"""Top-level graph assembly.
"""

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from market_research_team.agents.analytics.node import analytics_node
from market_research_team.agents.reporting.node import reporting_node
from market_research_team.agents.research.node import research_node
from market_research_team.config import settings
from market_research_team.guardrails import with_error_boundary
from market_research_team.state import AgentState
from market_research_team.supervisor.router import route_from_supervisor, supervisor_node


def _build_graph(checkpointer: BaseCheckpointSaver[Any] | None = None):
    builder = StateGraph(AgentState)
    builder.add_node(
        "supervisor",
        with_error_boundary("supervisor", supervisor_node, fallback_updates={"next": "FINISH"}),
    )
    builder.add_node("research", with_error_boundary("research", research_node))
    builder.add_node("analytics", with_error_boundary("analytics", analytics_node))
    builder.add_node("reporting", with_error_boundary("reporting", reporting_node))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "analytics": "analytics",
            "reporting": "reporting",
            END: END,
        },
    )
    builder.add_edge("research", "supervisor")
    builder.add_edge("analytics", "supervisor")
    builder.add_edge("reporting", "supervisor")

    return builder.compile(checkpointer=checkpointer)


# Exported for `langgraph.json` / `langgraph dev`: no checkpointer baked in,
# since the dev server (and LangGraph Platform generally) manages its own
# persistence for graphs it serves.
graph = _build_graph()


def build_production_graph(checkpointer: BaseCheckpointSaver[Any]):
    """Compile the graph with a real checkpointer, for standalone use outside
    `langgraph dev` (scripts, a future API wrapper) where nothing else is
    providing persistence."""

    return _build_graph(checkpointer=checkpointer)


def run_graph(
    initial_state: AgentState,
    *,
    compiled_graph: Any = None,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
) -> AgentState:
    """Safe entrypoint: invoke with a bounded recursion limit, and turn a
    `GraphRecursionError` into the same `error`-populated state shape the
    per-node error boundaries already produce, instead of letting it
    propagate as a raw exception up to the caller.
    """

    target_graph = compiled_graph if compiled_graph is not None else graph
    config: RunnableConfig = {"recursion_limit": recursion_limit or settings.recursion_limit}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}

    try:
        # `compiled_graph: Any` (deliberate — see above) makes `target_graph`'s
        # type partially unknown to the checker; the real object is always a
        # CompiledStateGraph with a normal `.invoke`.
        result = target_graph.invoke(initial_state, config=config)  # type: ignore[reportUnknownMemberType]
    except GraphRecursionError as exc:
        return {**initial_state, "error": f"Recursion limit reached: {exc}"}
    return cast(AgentState, result)
