"""Graph level: insights Analytics submits reach Reporting's draft."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as supervisor_router_module
from market_research_team.agents.supervisor.router import SupervisorRoute
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState, new_run_state


def _decide(state: AgentState) -> SupervisorRoute:
    if not state.get("research_findings"):
        return SupervisorRoute("research")
    if not state.get("analytics_results"):
        return SupervisorRoute("analytics")
    return SupervisorRoute("reporting" if not state.get("report_path") else "FINISH")


def test_submitted_insights_reach_the_report_draft(monkeypatch) -> None:
    drafted: list[dict[str, Any]] = []
    result = {"metric": "mean", "value": 35.0, "detail": "d", "inputs": [15.0, 55.0], "id": "r1"}
    insight = {"text": "The average seat price is $35.", "result_ids": ["r1"]}

    def _draft(objective, findings, results, llm, feedback=None, **kwargs):
        drafted.append(kwargs)
        return "# R"

    monkeypatch.setattr(
        research_node_module,
        "run_research_pipeline",
        lambda objective, focus=None, exclude=frozenset(): (
            [{"source": "a.md", "content": "Acme $55, Initech $15.", "relevance_score": 1.0}],
            1,
            1,
            [],
        ),
    )
    monkeypatch.setattr(
        analytics_node_module,
        "run_analytics_pipeline",
        lambda objective, findings: ([result], [insight]),
    )
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(reporting_node_module, "draft_report", _draft)
    monkeypatch.setattr(supervisor_router_module, "run_supervisor_decision", _decide)

    run_graph(
        new_run_state("Compare Acme and Initech"),
        compiled_graph=build_production_graph(InMemorySaver()),
        thread_id="handoff",
    )

    assert drafted and drafted[0]["insights"] == [insight]
