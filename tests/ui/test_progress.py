"""Live progress in the Gradio UI: step lines, the one-thread runner, and the
streaming path through `submit_objective` / `resolve_interrupt`."""

import threading
from typing import Any

import pytest
from gradio_app.app import (
    describe_step,
    progress_turn,
    resolve_interrupt,
    run_with_progress,
    submit_objective,
    with_steps,
)
from langchain_core.messages import AIMessage

from market_research_team.state import AgentState


def _final_state(**extra: Any) -> AgentState:
    state: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "FINISH",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
        "guardrail_events": [],
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


class _StreamingGraph:
    """Scripted `stream` output, shaped like LangGraph's
    `stream_mode=["updates", "values"]`."""

    def __init__(self, final: AgentState) -> None:
        self.final = final
        self.stream_calls = 0

    def stream(self, _input: object, config: object, stream_mode: list[str]):
        self.stream_calls += 1
        assert stream_mode == ["updates", "values"]
        yield "updates", {"input_guard": {"objective": "Compare Acme vs Globex pricing"}}
        yield "updates", {"research": {"messages": [AIMessage(content="Research complete: 3.")]}}
        yield "values", self.final

    def invoke(self, *_args: object, **_kwargs: object) -> AgentState:
        raise AssertionError("with on_step the graph must be streamed, not invoked")


def test_describe_step_prefers_the_nodes_own_message() -> None:
    update = {"messages": [AIMessage(content="Analytics complete: 4 metric(s).")]}

    assert describe_step("analytics", update) == "Analytics complete: 4 metric(s)."
    assert describe_step("input_guard", {"objective": "x"}) == "Objective checked."
    assert describe_step("research", {"error": "boom"}) == "research failed: boom"
    assert describe_step("mystery", {}) == "mystery finished."


def test_run_with_progress_yields_each_step_then_the_result_from_one_worker_thread() -> None:
    threads: set[str] = set()

    def run(on_step):
        for node in ("input_guard", "research"):
            threads.add(threading.current_thread().name)
            on_step(node, {})
        return "result"

    yields = [(list(steps), outcome) for steps, outcome in run_with_progress(run)]

    assert yields == [
        (["Objective checked."], None),
        (["Objective checked.", "research finished."], None),
        (["Objective checked.", "research finished."], "result"),
    ]
    assert threads == {"gradio-graph-run"}


def test_run_with_progress_reraises_a_crash_in_the_caller() -> None:
    def run(on_step):
        raise RuntimeError("worker crashed")

    with pytest.raises(RuntimeError, match="worker crashed"):
        list(run_with_progress(run))


def test_submit_objective_streams_steps_and_returns_the_usual_result() -> None:
    graph = _StreamingGraph(_final_state(report_path="/tmp/report.md"))
    steps: list[str] = []

    history, _thread, pending, _fig, approval, _obj, concluded = submit_objective(
        "Compare Acme vs Globex pricing",
        [],
        None,
        graph,
        on_step=lambda node, update: steps.append(describe_step(node, update)),
    )

    assert graph.stream_calls == 1
    assert steps == ["Objective checked.", "Research complete: 3."]
    assert pending is None and approval is False and concluded is True
    assert "report.md" in history[-1]["content"]


def test_discard_decision_resumes_and_ends_without_a_report() -> None:
    graph = _StreamingGraph(_final_state(report_discarded=True))

    history, pending, _fig, approval, concluded = resolve_interrupt(
        {"discard": True}, [], "t-1", graph, on_step=lambda node, update: None
    )

    assert history[0]["content"] == "Decision: {'discard': True}"
    assert history[-1]["content"] == "Report discarded."
    assert pending is None and approval is False and concluded is True


def test_with_steps_keeps_the_step_list_just_before_the_final_turn() -> None:
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "done"}]

    merged = with_steps(history, ["Objective checked."])

    assert merged[0] == history[0] and merged[-1] == history[-1]
    assert merged[1] == progress_turn(["Objective checked."], running=False)
    assert merged[1]["content"].startswith("**Run steps**")
    assert with_steps(history, []) == history
