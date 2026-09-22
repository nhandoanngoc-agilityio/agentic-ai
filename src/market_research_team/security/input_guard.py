"""Input guard node: the first node in the graph, so every surface (CLI,
`langgraph dev`, the frontend) gets the same validation before any LLM call."""

from typing import Any

from langchain_core.messages import AIMessage

from market_research_team.security.input_validation import validate_objective
from market_research_team.state import AgentState, GuardrailEvent


def input_guard_node(state: AgentState) -> dict[str, Any]:
    try:
        cleaned = validate_objective(state["objective"])
    except ValueError as exc:
        event: GuardrailEvent = {"layer": "input", "rule": "validate_objective", "detail": str(exc)}
        return {
            "error": f"Input rejected: {exc}",
            "next": "FINISH",
            "guardrail_events": [event],
            "messages": [AIMessage(content=f"Input rejected: {exc}", name="input_guard")],
        }
    return {"objective": cleaned}
