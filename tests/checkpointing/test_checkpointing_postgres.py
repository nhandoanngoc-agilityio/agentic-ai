"""Integration test for the Postgres checkpointer: proves state actually
persists across a real compiled graph invocation against a live Postgres
instance, mirroring test_checkpointing.py's SQLite coverage.

Skipped automatically unless DATABASE_URL points at a reachable Postgres —
never runs in CI or for developers without Postgres installed. See
docs/postgres_checkpointer.md for how to stand one up locally.
"""

import os
import uuid
from typing import Any

import pytest
from langgraph.types import Command

pytest.importorskip("psycopg", reason="psycopg not installed (pip install -e '.[prod]')")

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as supervisor_router_module
from market_research_team.checkpointing.store import get_checkpointer
from market_research_team.config import settings
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding, RouteDecision

_DATABASE_URL = os.environ.get("DATABASE_URL")


def _postgres_reachable(database_url: str | None) -> bool:
    if not database_url:
        return False
    import psycopg

    try:
        with psycopg.connect(database_url, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(_DATABASE_URL),
    reason="DATABASE_URL not set or Postgres not reachable -- see docs/postgres_checkpointer.md",
)


def _fake_research_pipeline(objective: str) -> tuple[list[ResearchFinding], int, int]:
    return [{"source": "mock", "content": f"finding for {objective}", "relevance_score": 0.9}], 1, 1


def _fake_analytics_pipeline(
    objective: str, findings: list[ResearchFinding]
) -> list[AnalyticsResult]:
    return [{"metric": "count", "value": float(len(findings)), "detail": "d"}]


def _fake_draft_report(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
    llm: Any,
    feedback: str | None = None,
) -> str:
    return "# Mock Report"


async def _fake_write_report_via_mcp(filename: str, content: str) -> str:
    return "reports/mock-report.md"


def _fake_supervisor_decision(state: AgentState) -> RouteDecision:
    if not state.get("research_findings"):
        return "research"
    if not state.get("analytics_results"):
        return "analytics"
    if not state.get("report_path"):
        return "reporting"
    return "FINISH"


@pytest.fixture(autouse=True)
def _stub_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research_node_module, "run_research_pipeline", _fake_research_pipeline)
    monkeypatch.setattr(analytics_node_module, "run_analytics_pipeline", _fake_analytics_pipeline)
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report)
    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _fake_write_report_via_mcp)
    monkeypatch.setattr(
        supervisor_router_module, "run_supervisor_decision", _fake_supervisor_decision
    )


@pytest.fixture(autouse=True)
def _use_real_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "database_url", _DATABASE_URL)


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def test_get_checkpointer_uses_postgres_when_database_url_set() -> None:
    from langgraph.checkpoint.postgres import PostgresSaver

    checkpointer = get_checkpointer()

    assert isinstance(checkpointer, PostgresSaver)


def test_postgres_checkpointer_persists_state_across_the_run() -> None:
    thread_id = f"pg-thread-{uuid.uuid4()}"
    checkpointer = get_checkpointer()
    compiled_graph = build_production_graph(checkpointer)

    result = run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id=thread_id)
    while "__interrupt__" in result:
        result = run_graph(
            Command(resume={"approved": True}), compiled_graph=compiled_graph, thread_id=thread_id
        )

    assert result["next"] == "FINISH"
    assert result["report_path"] == "reports/mock-report.md"

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    history = list(compiled_graph.get_state_history(config))
    assert len(history) > 1, "expected multiple checkpoints to have been recorded in Postgres"

    latest = compiled_graph.get_state(config)
    assert latest.values["report_path"] == "reports/mock-report.md"


def test_two_threads_do_not_share_checkpointed_state() -> None:
    thread_a = f"pg-thread-a-{uuid.uuid4()}"
    thread_b = f"pg-thread-b-{uuid.uuid4()}"
    checkpointer = get_checkpointer()
    compiled_graph = build_production_graph(checkpointer)

    for thread_id in (thread_a, thread_b):
        result = run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id=thread_id)
        while "__interrupt__" in result:
            result = run_graph(
                Command(resume={"approved": True}),
                compiled_graph=compiled_graph,
                thread_id=thread_id,
            )

    state_a = compiled_graph.get_state({"configurable": {"thread_id": thread_a}})
    state_b = compiled_graph.get_state({"configurable": {"thread_id": thread_b}})

    assert state_a.values["report_path"] == "reports/mock-report.md"
    assert state_b.values["report_path"] == "reports/mock-report.md"
    thread_id_a = state_a.config["configurable"]["thread_id"]
    thread_id_b = state_b.config["configurable"]["thread_id"]
    assert thread_id_a != thread_id_b
