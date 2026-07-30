import pytest

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.graph import graph
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding, RouteDecision
from market_research_team.supervisor import router as supervisor_router_module


def _fake_research_pipeline(objective: str) -> tuple[list[ResearchFinding], int, int]:
    findings: list[ResearchFinding] = [
        {
            "source": "mock://research-agent",
            "content": f"Mock research finding for objective: {objective!r}",
            "relevance_score": 0.9,
        }
    ]
    return findings, 1, 1


def _fake_analytics_pipeline(
    objective: str, findings: list[ResearchFinding]
) -> list[AnalyticsResult]:
    return [
        {
            "metric": "mock_finding_count",
            "value": float(len(findings)),
            "detail": "Mock analytics result for testing.",
        }
    ]


async def _fake_reporting_pipeline(
    objective: str, findings: list[ResearchFinding], results: list[AnalyticsResult]
) -> str:
    """Async on purpose: `reporting_node` wraps the real (async) pipeline in
    `asyncio.run(...)`, so the stub must also be awaitable."""

    return "reports/mock-report.md"


def _fake_supervisor_decision(state: AgentState) -> RouteDecision:
    """Reproduces the original deterministic research -> analytics -> reporting
    -> FINISH pass, so this graph-level test verifies topology/wiring only.
    Real handoff-decision logic is covered by test_supervisor_decision.py.
    """

    if not state.get("research_findings"):
        return "research"
    if not state.get("analytics_results"):
        return "analytics"
    if not state.get("report_path"):
        return "reporting"
    return "FINISH"


@pytest.fixture(autouse=True)
def _stub_agent_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the supervisor-routing test hermetic: no live LLM/vector-store calls."""

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


def test_supervisor_routes_through_all_agents_and_finishes() -> None:
    result = graph.invoke(_initial_state())

    assert result["next"] == "FINISH"
    assert len(result["research_findings"]) == 1
    assert len(result["analytics_results"]) == 1
    assert result["report_path"] == "reports/mock-report.md"


def test_supervisor_visits_agents_in_order() -> None:
    result = graph.invoke(_initial_state())

    agent_names = [m.name for m in result["messages"] if getattr(m, "name", None)]

    assert agent_names == [
        "supervisor",
        "research_agent",
        "supervisor",
        "analytics_agent",
        "supervisor",
        "reporting_agent",
        "supervisor",
    ]


def test_supervisor_can_hand_off_back_to_research_before_finishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the compiled graph itself supports bouncing between Research
    and Analytics (not just the isolated decision function in
    test_supervisor_decision.py) — the whole point of Day 7's handoff work.
    """

    decisions = iter(["research", "analytics", "research", "analytics", "reporting", "FINISH"])

    def _bouncing_decision(_state: AgentState) -> RouteDecision:
        return next(decisions)

    monkeypatch.setattr(supervisor_router_module, "run_supervisor_decision", _bouncing_decision)

    result = graph.invoke(_initial_state())

    agent_names = [m.name for m in result["messages"] if getattr(m, "name", None)]
    assert agent_names.count("research_agent") == 2
    assert agent_names.count("analytics_agent") == 2
    assert result["next"] == "FINISH"
