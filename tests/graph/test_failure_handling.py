"""Failures inside a real graph run end it cleanly and say what kind of
failure it was: no crash, no report written, the cause in state and audit."""

import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_research_team import guardrails as guardrails_module
from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as router_module
from market_research_team.agents.supervisor.router import SupervisorRoute
from market_research_team.config import settings
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState, new_run_state


class RateLimitError(Exception):
    """Named like the provider SDKs' rate-limit error."""


def _next_step(state: AgentState) -> SupervisorRoute:
    if state.get("error"):
        return SupervisorRoute("FINISH")
    if not state.get("research_findings"):
        return SupervisorRoute("research")
    if not state.get("analytics_results"):
        return SupervisorRoute("analytics")
    return SupervisorRoute("reporting" if not state.get("report_path") else "FINISH")


@pytest.fixture
def audit_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def _record(event: str, thread_id: str | None, **fields: Any) -> None:
        events.append({"event": event, **fields})

    monkeypatch.setattr(guardrails_module.audit, "record", _record)
    monkeypatch.setattr(router_module.audit, "record", _record)
    return events


@pytest.fixture(autouse=True)
def _stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        research_node_module,
        "run_research_pipeline",
        lambda objective, focus=None, exclude=frozenset(): (
            [{"source": "acme.md", "content": "Acme is $49.", "relevance_score": 1.0}],
            1,
            1,
            [],
        ),
    )
    monkeypatch.setattr(
        analytics_node_module,
        "run_analytics_pipeline",
        lambda objective, findings: [
            {"metric": "mean", "value": 49.0, "detail": "d", "entity": None, "inputs": [49.0]}
        ],
    )
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(reporting_node_module, "draft_report", lambda *a, **k: "# Report")
    monkeypatch.setattr(router_module, "run_supervisor_decision", _next_step)


def test_a_transient_failure_mid_run_ends_it_cleanly_and_is_labelled(
    monkeypatch: pytest.MonkeyPatch, audit_events: list[dict[str, Any]]
) -> None:
    def _rate_limited(objective: str, findings: list[Any]) -> None:
        raise RateLimitError("429 Too Many Requests")

    monkeypatch.setattr(analytics_node_module, "run_analytics_pipeline", _rate_limited)

    result = run_graph(
        new_run_state("Assess Acme pricing"),
        compiled_graph=build_production_graph(InMemorySaver()),
        thread_id="transient-failure",
    )

    assert result["next"] == "FINISH"
    assert result.get("report_path") is None
    assert result.get("error") == (
        "analytics failed (RateLimitError, transient): 429 Too Many Requests"
    )
    (failed,) = [
        e for e in audit_events if e["event"] == "node_latency" and e["outcome"] == "error"
    ]
    assert (failed["node"], failed["error_type"], failed["transient"]) == (
        "analytics",
        "RateLimitError",
        True,
    )
    (finished,) = [e for e in audit_events if e["event"] == "run_finished"]
    assert "transient" in finished["error"]


def test_an_mcp_write_that_times_out_after_approval_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, audit_events: list[dict[str, Any]]
) -> None:
    async def _stalled_server() -> list[Any]:
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(reporting_node_module, "load_reporting_tools", _stalled_server)
    monkeypatch.setattr(settings, "mcp_write_timeout_seconds", 0.05)
    graph = build_production_graph(InMemorySaver())

    paused = run_graph(
        new_run_state("Assess Acme pricing"), compiled_graph=graph, thread_id="mcp-timeout"
    )
    assert "__interrupt__" in paused
    result = run_graph(
        Command(resume={"approved": True}), compiled_graph=graph, thread_id="mcp-timeout"
    )

    assert result.get("report_path") is None
    assert (result.get("error") or "").startswith("report_review failed (TimeoutError, transient)")
    assert result["next"] == "FINISH"
