"""Integration test for the SQLite checkpointer: proves state actually
persists across a real compiled graph invocation, not just that
`get_checkpointer()` returns some object."""

from pathlib import Path
from typing import Any

import pytest

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.checkpointing.store import get_checkpointer
from market_research_team.config import settings
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding, RouteDecision
from market_research_team.supervisor import router as supervisor_router_module


def _fake_research_pipeline(objective: str) -> tuple[list[ResearchFinding], int, int]:
    return [{"source": "mock", "content": f"finding for {objective}", "relevance_score": 0.9}], 1, 1


def _fake_analytics_pipeline(
    objective: str, findings: list[ResearchFinding]
) -> list[AnalyticsResult]:
    return [{"metric": "count", "value": float(len(findings)), "detail": "d"}]


async def _fake_reporting_pipeline(
    objective: str, findings: list[ResearchFinding], results: list[AnalyticsResult]
) -> str:
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
    monkeypatch.setattr(
        reporting_node_module, "run_reporting_pipeline", _fake_reporting_pipeline
    )
    monkeypatch.setattr(
        supervisor_router_module, "run_supervisor_decision", _fake_supervisor_decision
    )


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def test_get_checkpointer_uses_sqlite_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(settings, "checkpoint_db_path", db_path)
    monkeypatch.setattr(settings, "database_url", None)

    get_checkpointer()

    assert db_path.exists()


def test_sqlite_checkpointer_persists_state_across_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "checkpoint_db_path", tmp_path / "checkpoints.sqlite")

    checkpointer = get_checkpointer()
    compiled_graph = build_production_graph(checkpointer)

    result = run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id="thread-abc")

    assert result["next"] == "FINISH"
    assert result["report_path"] == "reports/mock-report.md"

    config: dict[str, Any] = {"configurable": {"thread_id": "thread-abc"}}
    history = list(compiled_graph.get_state_history(config))
    assert len(history) > 1, "expected multiple checkpoints to have been recorded"

    latest = compiled_graph.get_state(config)
    assert latest.values["report_path"] == "reports/mock-report.md"


def test_two_threads_do_not_share_checkpointed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "checkpoint_db_path", tmp_path / "checkpoints.sqlite")

    checkpointer = get_checkpointer()
    compiled_graph = build_production_graph(checkpointer)

    run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id="thread-a")
    run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id="thread-b")

    state_a = compiled_graph.get_state({"configurable": {"thread_id": "thread-a"}})
    state_b = compiled_graph.get_state({"configurable": {"thread_id": "thread-b"}})

    assert state_a.values["report_path"] == "reports/mock-report.md"
    assert state_b.values["report_path"] == "reports/mock-report.md"
    thread_a = state_a.config["configurable"]["thread_id"]
    thread_b = state_b.config["configurable"]["thread_id"]
    assert thread_a != thread_b
