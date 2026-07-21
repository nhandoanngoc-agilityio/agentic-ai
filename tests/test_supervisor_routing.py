from market_research_team.graph import graph
from market_research_team.state import AgentState


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
