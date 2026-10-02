"""Graph runs used by evals answer the human-approval interrupt themselves."""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from market_research_team import graph as graph_module
from market_research_team.evaluation import graph_runs
from market_research_team.security import audit


def test_run_graph_with_decision_resumes_with_the_given_decision(monkeypatch):
    calls: list[object] = []

    def fake_run_graph(state, *, compiled_graph=None, thread_id=None, **_):
        calls.append(state)
        if isinstance(state, Command):
            return {"report_path": "reports/x.md"}
        return {"__interrupt__": ["pending"]}

    monkeypatch.setattr(graph_module, "run_graph", fake_run_graph)

    final = graph_runs.run_graph_with_decision(
        "Assess Acme pricing", thread_id="eval-x-r0", decision={"approved": True}
    )

    assert final["report_path"] == "reports/x.md"
    assert isinstance(calls[1], Command) and calls[1].resume == {"approved": True}


def test_run_graph_with_decision_stops_after_bounded_resumes(monkeypatch):
    calls: list[object] = []

    def always_interrupt(state, **_):
        calls.append(state)
        return {"__interrupt__": ["pending"]}

    monkeypatch.setattr(graph_module, "run_graph", always_interrupt)

    final = graph_runs.run_graph_with_decision(
        "Assess Acme pricing", thread_id="eval-x-r0", decision={"approved": False}
    )

    assert final.get("__interrupt__")
    assert len(calls) == 1 + graph_runs.MAX_APPROVAL_ROUNDS


def test_audit_thread_id_is_visible_inside_graph_nodes():
    # Cost per run groups llm_usage events by thread id, which audit reads from
    # LangGraph's run config inside nodes. Pin that it works.
    class _State(TypedDict):
        seen: str | None

    def node(_state: _State) -> dict:
        return {"seen": audit.current_thread_id()}

    builder = StateGraph(_State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    compiled = builder.compile(checkpointer=InMemorySaver())

    out = compiled.invoke({"seen": None}, config={"configurable": {"thread_id": "eval-t-r0"}})

    assert out["seen"] == "eval-t-r0"


def test_run_graph_with_outcome_counts_approval_rounds(monkeypatch):
    def fake_run_graph(state, **_):
        if isinstance(state, Command):
            return {"report_path": "reports/x.md"}
        return {"__interrupt__": ["pending"], "report_path": None}

    monkeypatch.setattr(graph_module, "run_graph", fake_run_graph)

    outcome = graph_runs.run_graph_with_outcome(
        "o", thread_id="eval-x-r0", decision={"approved": True}
    )

    assert outcome.approval_rounds == 1
    assert outcome.wrote_before_approval is False
    assert outcome.state["report_path"] == "reports/x.md"


def test_run_graph_with_outcome_flags_a_write_without_asking(monkeypatch):
    monkeypatch.setattr(graph_module, "run_graph", lambda state, **_: {"report_path": "r.md"})

    outcome = graph_runs.run_graph_with_outcome("o", thread_id="eval-x-r0", decision={})

    assert outcome.approval_rounds == 0
    assert outcome.wrote_before_approval is True
