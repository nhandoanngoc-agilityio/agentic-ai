"""Cross-cutting production guardrail: per-node error boundaries.

Applied when nodes are registered in `graph.py`, rather than inside each
node module, so every node gets the same containment behavior without the
node's own code needing to know about it.
"""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from market_research_team.state import AgentState

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], dict[str, Any]]


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
        try:
            return node_fn(state)
        except Exception as exc:
            logger.exception("Node %r failed", node_name)
            update: dict[str, Any] = {
                "error": f"{node_name} failed: {exc}",
                "messages": [
                    AIMessage(content=f"{node_name} failed: {exc}", name=node_name)
                ],
            }
            if fallback_updates:
                update.update(fallback_updates)
            return update

    return _wrapped
