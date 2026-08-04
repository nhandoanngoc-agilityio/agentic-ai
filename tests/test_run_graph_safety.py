"""Tests for the `run_graph` safe entrypoint: recursion-limit handling."""

import pytest

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.graph import run_graph
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


def test_run_graph_completes_normally_within_default_recursion_limit() -> None:
    result = run_graph(_initial_state())

    assert result["next"] == "FINISH"
    assert result.get("error") is None


def test_run_graph_catches_graph_recursion_error_and_returns_error_state() -> None:
    result = run_graph(_initial_state(), recursion_limit=2)

    assert result.get("error") is not None
    assert "Recursion limit reached" in result["error"]
