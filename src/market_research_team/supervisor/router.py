"""Supervisor node: decides which sub-agent runs next, including handoffs
back to Research or Analytics based on how the run is progressing."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from pydantic import BaseModel, Field

from market_research_team.config import settings
from market_research_team.state import AgentState, RouteDecision

_MAX_ROUTING_VISITS = 6

_ROUTE_ANNOUNCEMENT: dict[RouteDecision, str] = {
    "research": "Routing to the Research Agent.",
    "analytics": "Routing to the Analytics Agent.",
    "reporting": "Routing to the Reporting Agent.",
    "FINISH": "All sub-agents have reported back. Ending run.",
}

_SYSTEM_PROMPT = (
    "You are the supervisor for a market and competitor research team made "
    "of a Research Agent, an Analytics Agent, and a Reporting Agent. Given "
    "the objective and a summary of what's been gathered so far, decide "
    "which agent should act next. Send work back to Research if Analytics "
    "would benefit from more source material; send work to Analytics once "
    "there's something new worth analyzing; move to Reporting once there's "
    "enough analysis to write up. Only choose from the allowed options "
    "listed below."
)


class _SupervisorDecision(BaseModel):
    next: RouteDecision = Field(description="Which node should run next.")


def _supervisor_visit_count(state: AgentState) -> int:
    return sum(1 for message in state["messages"] if getattr(message, "name", None) == "supervisor")


def _ask_llm_for_route(
    state: AgentState, llm: BaseChatModel, allowed: tuple[RouteDecision, ...]
) -> RouteDecision:
    context = (
        f"Objective: {state['objective']}\n"
        f"Research findings so far: {len(state.get('research_findings', []))}\n"
        f"Analytics results so far: {len(state.get('analytics_results', []))}\n"
        f"Allowed next steps: {', '.join(allowed)}"
    )
    structured_llm = llm.with_structured_output(_SupervisorDecision)
    decision = structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=context)]
    )
    return decision.next if decision.next in allowed else allowed[0]


def decide_next_step(state: AgentState, llm: BaseChatModel) -> RouteDecision:
    """Decide the next node, allowing handoffs back to Research or Analytics.

    Hard prerequisites are enforced in code rather than left to the LLM:
    Analytics needs at least one research pass to have something to
    analyze, and once a report has been written there's nothing left to
    orchestrate. Within those constraints, the LLM picks whether to
    proceed or hand back for another round — capped by
    `_MAX_ROUTING_VISITS` so a bad decision can't loop forever, and
    clamped to the currently allowed options so a malformed answer can't
    route somewhere invalid. Falls back to the safest allowed option if
    the LLM call itself fails.
    """

    if not state.get("research_findings"):
        return "research"

    if state.get("report_path"):
        return "FINISH"

    if _supervisor_visit_count(state) >= _MAX_ROUTING_VISITS:
        return "analytics" if not state.get("analytics_results") else "reporting"

    allowed: tuple[RouteDecision, ...] = (
        ("research", "analytics")
        if not state.get("analytics_results")
        else ("research", "analytics", "reporting")
    )

    try:
        return _ask_llm_for_route(state, llm, allowed)
    except Exception:
        return allowed[-1]


def run_supervisor_decision(state: AgentState) -> RouteDecision:
    """Production wiring: the real Claude model driving `decide_next_step`."""

    llm = ChatAnthropic(model=settings.query_rewrite_model)
    return decide_next_step(state, llm)


def supervisor_node(state: AgentState) -> dict[str, Any]:
    next_step = run_supervisor_decision(state)
    return {
        "next": next_step,
        "messages": [AIMessage(content=_ROUTE_ANNOUNCEMENT[next_step], name="supervisor")],
    }


def route_from_supervisor(state: AgentState) -> str:
    """Conditional-edge mapper: AgentState["next"] -> graph node name or END."""
    next_step = state["next"]
    return END if next_step == "FINISH" else next_step
