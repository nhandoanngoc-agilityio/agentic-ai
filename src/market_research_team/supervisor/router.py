"""Supervisor node: decides which sub-agent runs next."""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END

from market_research_team.state import AgentState, RouteDecision

_ROUTE_ANNOUNCEMENT: dict[RouteDecision, str] = {
    "research": "Routing to the Research Agent.",
    "analytics": "Routing to the Analytics Agent.",
    "reporting": "Routing to the Reporting Agent.",
    "FINISH": "All sub-agents have reported back. Ending run.",
}


def _decide_next(state: AgentState) -> RouteDecision:
    if not state.get("research_findings"):
        return "research"
    if not state.get("analytics_results"):
        return "analytics"
    if not state.get("report_path"):
        return "reporting"
    return "FINISH"


def supervisor_node(state: AgentState) -> dict[str, Any]:
    next_step = _decide_next(state)
    return {
        "next": next_step,
        "messages": [AIMessage(content=_ROUTE_ANNOUNCEMENT[next_step], name="supervisor")],
    }


def route_from_supervisor(state: AgentState) -> str:
    """Conditional-edge mapper: AgentState["next"] -> graph node name or END."""
    next_step = state["next"]
    return END if next_step == "FINISH" else next_step
