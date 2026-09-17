from market_research_team.state import AgentState, RouteDecision


def test_agent_state_accepts_expected_keys() -> None:
    state: AgentState = {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }

    assert state["next"] in RouteDecision.__args__
    assert state["report_path"] is None
