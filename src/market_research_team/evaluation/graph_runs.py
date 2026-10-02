"""Real graph runs for evals: an in-memory checkpointer plus a scripted answer
to the reporting agent's human-approval interrupt (evals have no human).

Every run gets an explicit `thread_id` starting with `eval-`, which is how
`metrics.py` groups latency and token usage per run.
"""

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_research_team import graph as graph_module
from market_research_team.state import AgentState

# A reviewer can reject up to `_MAX_REVIEW_ROUNDS` (3) drafts; never loop forever.
MAX_APPROVAL_ROUNDS = 3


def pipeline_state(objective: str) -> AgentState:
    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


@dataclass
class GraphRunOutcome:
    state: AgentState
    # How many approval interrupts were answered.
    approval_rounds: int
    # True when the first state already had a report path, i.e. the graph
    # wrote without pausing for approval.
    wrote_before_approval: bool


def run_graph_with_outcome(
    objective: str, *, thread_id: str, decision: dict[str, Any]
) -> GraphRunOutcome:
    """Run the graph, answer every approval interrupt with `decision`, and
    report how the approval step behaved.

    Uses `graph_module.run_graph` (not a direct import) so tests can
    monkeypatch it. The final state still carries `__interrupt__` if the
    graph kept asking after `MAX_APPROVAL_ROUNDS` answers.
    """

    compiled = graph_module.build_production_graph(InMemorySaver())
    state = graph_module.run_graph(
        pipeline_state(objective), compiled_graph=compiled, thread_id=thread_id
    )
    wrote_before_approval = bool(state.get("report_path"))
    rounds = 0
    for _ in range(MAX_APPROVAL_ROUNDS):
        if not state.get("__interrupt__"):
            break
        rounds += 1
        state = graph_module.run_graph(
            Command(resume=decision), compiled_graph=compiled, thread_id=thread_id
        )
    return GraphRunOutcome(state, rounds, wrote_before_approval)


def run_graph_with_decision(
    objective: str, *, thread_id: str, decision: dict[str, Any]
) -> AgentState:
    """Run the graph and answer every approval interrupt with `decision`."""

    return run_graph_with_outcome(objective, thread_id=thread_id, decision=decision).state
