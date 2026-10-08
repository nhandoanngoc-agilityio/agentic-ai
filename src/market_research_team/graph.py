"""Top-level graph assembly."""

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from market_research_team.agents.analytics.node import analytics_node
from market_research_team.agents.planner.node import planner_node
from market_research_team.agents.reporting.node import (
    report_review_node,
    reporting_node,
    route_after_draft,
    route_after_review,
)
from market_research_team.agents.research.node import research_node
from market_research_team.agents.supervisor.router import route_from_supervisor, supervisor_node
from market_research_team.caching.response_cache import get_cached_response
from market_research_team.config import settings
from market_research_team.guardrails import with_error_boundary
from market_research_team.observability import flush as flush_traces
from market_research_team.observability import (
    get_langfuse_handler,
    take_run_usage,
    tracing_enabled,
)
from market_research_team.security import audit
from market_research_team.security.input_guard import input_guard_node
from market_research_team.security.input_validation import validate_objective
from market_research_team.state import AgentState
from market_research_team.versioning import trace_metadata

logger = logging.getLogger(__name__)

# Called after each node finishes: (node name, the partial state update it returned).
StepCallback = Callable[[str, dict[str, Any]], None]


def _build_graph(checkpointer: BaseCheckpointSaver[Any] | None = None):
    builder = StateGraph(AgentState)
    builder.add_node(
        "supervisor",
        with_error_boundary("supervisor", supervisor_node, fallback_updates={"next": "FINISH"}),
    )
    builder.add_node("research", with_error_boundary("research", research_node))
    builder.add_node("analytics", with_error_boundary("analytics", analytics_node))
    # Reporting is two nodes: `reporting` drafts, `report_review` holds the
    # human-approval interrupt and the write. Splitting them means resuming the
    # interrupt never re-drafts -- see `agents/reporting/node.py`.
    builder.add_node("reporting", with_error_boundary("reporting", reporting_node))
    builder.add_node("report_review", with_error_boundary("report_review", report_review_node))
    # Input guardrail runs first so CLI, `langgraph dev` and the frontend all
    # get the same validation; a rejection sets `error`, which the supervisor
    # turns into FINISH without calling the LLM.
    builder.add_node(
        "input_guard",
        with_error_boundary("input_guard", input_guard_node, fallback_updates={"next": "FINISH"}),
    )

    # The planner decomposes the objective into sub-questions once, before the
    # supervisor's first decision; a planning failure falls back to a
    # one-item plan inside the node, so it never ends the run.
    builder.add_node("planner", with_error_boundary("planner", planner_node))

    builder.add_edge(START, "input_guard")
    builder.add_edge("input_guard", "planner")
    builder.add_edge("planner", "supervisor")
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
    builder.add_conditional_edges(
        "reporting",
        route_after_draft,
        {"report_review": "report_review", "supervisor": "supervisor"},
    )
    builder.add_conditional_edges(
        "report_review",
        route_after_review,
        {"reporting": "reporting", "supervisor": "supervisor"},
    )

    return builder.compile(checkpointer=checkpointer)


# Exported for `langgraph.json` / `langgraph dev`: no checkpointer baked in,
# since the dev server (and LangGraph Platform generally) manages its own
# persistence for graphs it serves.
graph = _build_graph()
if tracing_enabled():
    # Baked into the compiled graph (rather than passed per-invoke) so
    # `langgraph dev` traces too -- Studio calls `graph.invoke`/`.stream`
    # directly and never goes through `run_graph`.
    graph = graph.with_config({"callbacks": [get_langfuse_handler()]})


def build_production_graph(checkpointer: BaseCheckpointSaver[Any]):
    """Compile the graph with a real checkpointer, for standalone use outside
    `langgraph dev` (scripts, a future API wrapper) where nothing else is
    providing persistence."""

    return _build_graph(checkpointer=checkpointer)


def _apply_response_cache(initial_state: AgentState | Command[Any]) -> AgentState | Command[Any]:
    """On a fresh (non-resume) run, check the opt-in response cache for an
    exact match on the validated objective. A hit populates
    `research_findings`/`analytics_results` and sets `from_response_cache`
    so `decide_next_step` skips straight to Reporting -- see
    `caching/response_cache.py`. Never raises: a bad objective here is left
    for `input_guard_node` to reject normally.
    """

    if not settings.response_cache_enabled or not isinstance(initial_state, dict):
        return initial_state

    try:
        objective = validate_objective(initial_state["objective"])
    except Exception:
        return initial_state

    cached = get_cached_response(objective, settings.vectorstore_dir)
    if cached is None:
        return initial_state

    findings, results = cached
    return {
        **initial_state,
        "objective": objective,
        "research_findings": findings,
        "analytics_results": results,
        "from_response_cache": True,
    }


def run_graph(
    initial_state: AgentState | Command[Any],
    *,
    compiled_graph: Any = None,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
    on_step: StepCallback | None = None,
) -> AgentState:
    """Safe entrypoint: invoke with a bounded recursion limit, and turn a
    `GraphRecursionError` into the same `error`-populated state shape the
    per-node error boundaries already produce, instead of letting it
    propagate as a raw exception up to the caller.

    `initial_state` may also be a `Command` (e.g. `Command(resume=...)`) to
    resume a run paused on a human-approval interrupt (see
    `agents/reporting/node.py`) — this requires `compiled_graph` to carry a
    real checkpointer and the same `thread_id` the paused run used. The
    returned state includes an `__interrupt__` key when the run pauses
    again rather than reaching `FINISH`.

    `on_step`, when given, is called after each node with its name and update
    (the Gradio UI uses it to show progress). The graph is then streamed rather
    than invoked; the returned state is the same either way.
    """

    initial_state = _apply_response_cache(initial_state)
    target_graph = compiled_graph if compiled_graph is not None else graph
    config: RunnableConfig = {"recursion_limit": recursion_limit or settings.recursion_limit}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    if tracing_enabled():
        config["callbacks"] = [get_langfuse_handler()]

    objective = initial_state.get("objective") if isinstance(initial_state, dict) else None
    start = time.perf_counter()
    try:
        if tracing_enabled():
            from langfuse import propagate_attributes

            version_metadata = trace_metadata()
            if objective:
                version_metadata["objective"] = objective
            with propagate_attributes(
                trace_name="market-research-run",
                session_id=thread_id,
                version=version_metadata["agent_version"],
                tags=["market-research-team"],
                metadata=version_metadata,
            ):
                result = _execute(target_graph, initial_state, config, on_step)
        else:
            result = _execute(target_graph, initial_state, config, on_step)
    except GraphRecursionError as exc:
        take_run_usage(thread_id)  # the run ended without a run_finished record
        _record_run_latency(thread_id, objective, start, "recursion_limit")
        state = _state_at_failure(target_graph, initial_state, config)
        return cast(AgentState, {**state, "error": f"Recursion limit reached: {exc}"})
    except Exception:
        take_run_usage(thread_id)
        _record_run_latency(thread_id, objective, start, "exception")
        raise
    finally:
        flush_traces()

    outcome = "interrupted" if result.get("__interrupt__") else "finished"
    _record_run_latency(thread_id, objective, start, outcome)
    return cast(AgentState, result)


def _state_at_failure(
    target_graph: Any, initial_state: AgentState | Command[Any], config: RunnableConfig
) -> dict[str, Any]:
    """The state to report a failed invocation with.

    A fresh run reports its input. A resume (`Command`) has no state of its
    own, so it reports the thread's last checkpoint, or `{}` when there is
    no checkpointer to read.
    """

    if isinstance(initial_state, dict):
        return dict(initial_state)
    try:
        return dict(target_graph.get_state(config).values)  # type: ignore[reportUnknownMemberType]
    except Exception:
        return {}


def _execute(
    target_graph: Any,
    initial_state: AgentState | Command[Any],
    config: RunnableConfig,
    on_step: StepCallback | None,
) -> dict[str, Any]:
    """`invoke`, or -- with `on_step` -- a stream that reports each node and
    rebuilds what `invoke` returns: the final state, plus `__interrupt__` when
    the run paused for approval."""

    if on_step is None:
        # `compiled_graph: Any` (deliberate, see `run_graph`) leaves the type partly unknown.
        return target_graph.invoke(initial_state, config=config)  # type: ignore[reportUnknownMemberType]

    final: dict[str, Any] = {}
    interrupts: list[Any] = []
    for mode, chunk in target_graph.stream(  # type: ignore[reportUnknownMemberType]
        initial_state, config=config, stream_mode=["updates", "values"]
    ):
        if mode == "values":
            final = chunk
            continue
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts.extend(update)
                continue
            try:
                on_step(node, update or {})
            except Exception:
                # Progress display must never break a run.
                logger.exception("on_step callback failed for node %r", node)
    if interrupts:
        final = {**final, "__interrupt__": interrupts}
    return final


def _record_run_latency(
    thread_id: str | None, objective: str | None, start: float, outcome: str
) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    audit.record(
        "run_latency", thread_id, objective=objective, duration_ms=duration_ms, outcome=outcome
    )
