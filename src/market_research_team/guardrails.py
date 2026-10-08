"""Cross-cutting production guardrail: per-node error boundaries.

Applied when nodes are registered in `graph.py`, rather than inside each
node module, so every node gets the same containment behavior without the
node's own code needing to know about it.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from market_research_team.security import audit
from market_research_team.state import AgentState

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], dict[str, Any]]

# Exception classes (by name, anywhere in the MRO) that mean "the call might
# succeed if repeated": provider SDKs have already retried these with backoff
# before they reach a node (`llm_max_retries`), so here they are recorded, not
# retried again. Matching by name keeps provider packages out of this module.
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
    }
)


def is_transient(exc: BaseException) -> bool:
    """True for timeouts, connection failures, rate limits and provider 5xx."""

    return any(cls.__name__ in _TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__)


# Written into `error` by `describe_failure`; the failure harvester reads it.
TRANSIENT_MARK = ", transient)"


def describe_failure(node_name: str, exc: BaseException) -> str:
    """`"<node> failed (<ExceptionType>[, transient]): <message>"` -- the type
    and the transient flag let a reader (or the failure harvester) tell a rate
    limit apart from a bug without the traceback."""

    marker = TRANSIENT_MARK if is_transient(exc) else ")"
    return f"{node_name} failed ({type(exc).__name__}{marker}: {exc}"


def with_error_boundary(
    node_name: str,
    node_fn: NodeFn,
    *,
    fallback_updates: dict[str, Any] | None = None,
):
    """Wrap a node so an unexpected exception degrades the run instead of crashing it.

    The exception is recorded into `state["error"]`. For research/analytics/
    reporting, that's enough: they always route back to the supervisor,
    whose `decide_next_step` checks `error` first and ends the run rather
    than retrying a node that just failed. The supervisor node itself has
    no "next turn" to catch its own failure, so it's registered with
    `fallback_updates={"next": "FINISH"}` to end the run directly.

    No explicit return-type annotation and no `functools.wraps` here on
    purpose: both erase the nested function's literal `(state: AgentState)`
    signature down to a nameless `Callable`, which LangGraph's `add_node`
    protocol then rejects — the same node function form that works fine
    passed directly stops satisfying the protocol once it's been through
    an annotated wrapper.
    """

    def _wrapped(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = node_fn(state)
        except GraphBubbleUp:
            # LangGraph's own control flow (interrupt(), Send, etc.) is
            # implemented as an exception that must reach the runtime
            # unchanged -- a bare `except Exception` below would otherwise
            # swallow a human-in-the-loop interrupt and mistake it for a
            # node crash. Still timed and recorded (as "interrupted") since
            # real work (e.g. drafting) ran before the pause.
            _record_node_latency(node_name, start, "interrupted")
            raise
        except Exception as exc:
            _record_node_latency(
                node_name,
                start,
                "error",
                error_type=type(exc).__name__,
                transient=is_transient(exc),
            )
            logger.exception("Node %r failed", node_name)
            message = describe_failure(node_name, exc)
            update: dict[str, Any] = {
                "error": message,
                "messages": [AIMessage(content=message, name=node_name)],
            }
            if fallback_updates:
                update.update(fallback_updates)
            return update
        else:
            _record_node_latency(node_name, start, "ok")
            return result

    return _wrapped


def _record_node_latency(node_name: str, start: float, outcome: str, **error: Any) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    audit.record(
        "node_latency",
        audit.current_thread_id(),
        node=node_name,
        duration_ms=duration_ms,
        outcome=outcome,
        **error,
    )
