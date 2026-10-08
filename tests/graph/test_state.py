from market_research_team.state import AgentState, RouteDecision, new_run_state


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


def test_new_run_state_sets_every_state_field() -> None:
    """Adding a field to `AgentState` without resetting it here fails this test,
    so a fresh run never inherits a value it did not set."""

    state = new_run_state("Assess Acme pricing")

    assert set(state) == set(AgentState.__annotations__)
    assert state["objective"] == "Assess Acme pricing"
    assert state.get("supervisor_visits") == 0
    assert state.get("guardrail_events") == []
