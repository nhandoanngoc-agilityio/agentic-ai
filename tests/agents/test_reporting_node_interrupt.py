"""Tests for the Reporting Agent's human-in-the-loop approval gate: the
write to disk must pause for review, a rejection can redirect the next
draft via feedback, and repeated rejection ends the run without writing.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.reporting.node import report_filename
from market_research_team.config import settings
from market_research_team.state import AgentState

_OBJECTIVE = "Assess competitor pricing strategy"


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "reporting",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def _fake_draft_report_recording(feedback_log: list[str | None]):
    def _draft(
        objective: str,
        findings: list[Any],
        results: list[Any],
        llm: Any,
        feedback: str | None = None,
    ) -> str:
        feedback_log.append(feedback)
        return f"# Draft (feedback={feedback})"

    return _draft


async def _fake_write_report_via_mcp(filename: str, content: str) -> str:
    return f"reports/{filename}"


@pytest.fixture(autouse=True)
def _stub_llm_and_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the interrupt-loop behavior from the real LLM and MCP server."""

    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _fake_write_report_via_mcp)


def _compiled_graph():
    """Draft + review wired as in `graph.py`, with the supervisor replaced by END."""

    builder = StateGraph(AgentState)
    builder.add_node("reporting", reporting_node_module.reporting_node)
    builder.add_node("report_review", reporting_node_module.report_review_node)
    builder.add_edge(START, "reporting")
    builder.add_conditional_edges(
        "reporting",
        reporting_node_module.route_after_draft,
        {"report_review": "report_review", "supervisor": END},
    )
    builder.add_conditional_edges(
        "report_review",
        reporting_node_module.route_after_review,
        {"reporting": "reporting", "supervisor": END},
    )
    return builder.compile(checkpointer=InMemorySaver())


def test_reporting_node_pauses_before_writing_and_writes_on_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report_recording([]))

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t1"}}

    paused = graph.invoke(_initial_state(), config=config)
    assert "__interrupt__" in paused
    interrupt_value = paused["__interrupt__"][0].value
    assert interrupt_value["action"] == "write_report"
    assert interrupt_value["filename"] == report_filename(_OBJECTIVE, "t1")
    assert interrupt_value["attempt"] == 1

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["report_path"] == f"reports/{report_filename(_OBJECTIVE, 't1')}"
    assert result.get("error") is None
    assert "__interrupt__" not in result


def test_reporting_node_populates_response_cache_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report_recording([]))
    calls: list[tuple[Any, Any, Any, Any]] = []
    monkeypatch.setattr(
        reporting_node_module,
        "put_cached_response",
        lambda objective, findings, results, persist_dir, guardrail_events=None: calls.append(
            (objective, findings, results, persist_dir)
        ),
    )

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t-cache"}}
    graph.invoke(_initial_state(), config=config)

    assert len(calls) == 1
    objective, findings, results, _persist_dir = calls[0]
    assert objective == "Assess competitor pricing strategy"
    assert findings == []
    assert results == []


def test_reporting_node_redrafts_with_feedback_after_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_log: list[str | None] = []
    monkeypatch.setattr(
        reporting_node_module, "draft_report", _fake_draft_report_recording(feedback_log)
    )

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t2"}}

    graph.invoke(_initial_state(), config=config)
    second = graph.invoke(
        Command(resume={"approved": False, "feedback": "Add a pricing table."}), config=config
    )

    assert "__interrupt__" in second
    assert second["__interrupt__"][0].value["attempt"] == 2
    assert second["__interrupt__"][0].value["content"] == "# Draft (feedback=Add a pricing table.)"

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["report_path"] is not None
    # Drafting is its own node, so resuming the review never re-drafts:
    # exactly one draft per round, the second carrying the feedback.
    assert feedback_log == [None, "Add a pricing table."]


def test_reporting_node_ends_run_after_max_review_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report_recording([]))

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t3"}}
    max_rounds = settings.run_policy.max_review_rounds

    result = graph.invoke(_initial_state(), config=config)
    for round_num in range(1, max_rounds + 1):
        assert "__interrupt__" in result, f"expected a pause on review round {round_num}"
        assert result["__interrupt__"][0].value["attempt"] == round_num
        result = graph.invoke(
            Command(resume={"approved": False, "feedback": "still not good"}), config=config
        )

    assert "__interrupt__" not in result
    assert result.get("report_path") is None
    assert result.get("error") is not None
    assert f"{max_rounds} review rounds" in result["error"]


def test_reporting_node_discards_immediately_without_further_redraft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_calls: list[str | None] = []
    monkeypatch.setattr(
        reporting_node_module, "draft_report", _fake_draft_report_recording(draft_calls)
    )

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t4"}}

    paused = graph.invoke(_initial_state(), config=config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["attempt"] == 1

    result = graph.invoke(Command(resume={"approved": False, "discard": True}), config=config)

    assert "__interrupt__" not in result
    assert result.get("report_path") is None
    assert result.get("error") is None
    # The discard branch's contract with the supervisor: a discard sets neither
    # `report_path` nor `error`, so `report_discarded` is the only signal
    # `decide_next_step` has to end the run instead of routing back to reporting.
    # (tests/test_graph_discard_terminates.py proves that end-to-end on the real graph.)
    assert result.get("report_discarded") is True
    # One draft only: the discard never triggers a redraft round.
    assert draft_calls == [None]


def test_approved_draft_is_exactly_what_gets_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: drafting used to live in the interrupting node, so resuming
    after approval re-ran the LLM and wrote a different, unreviewed draft. A
    non-deterministic fake LLM makes that visible."""

    calls = iter(range(100))
    monkeypatch.setattr(
        reporting_node_module,
        "draft_report",
        lambda objective, findings, results, llm, feedback=None: f"# Draft #{next(calls)}",
    )
    written: list[str] = []

    async def _recording_write(filename: str, content: str) -> str:
        written.append(content)
        return f"reports/{filename}"

    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _recording_write)

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t-exact"}}

    paused = graph.invoke(_initial_state(), config=config)
    paused = graph.invoke(Command(resume={"approved": False, "feedback": "shorter"}), config=config)
    shown = paused["__interrupt__"][0].value["content"]
    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert written == [shown]
    assert shown == "# Draft #1"
    assert result["report_review_round"] == 0
    assert result.get("report_feedback") is None


async def test_approval_writes_even_when_an_event_loop_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the review node wrote via `asyncio.run`, which raises when
    the graph is driven from code that already has a running loop (an async
    handler, a notebook). A sync `graph.invoke` inside this async test runs the
    node on the loop's thread, reproducing that."""

    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report_recording([]))
    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t-loop"}}

    graph.invoke(_initial_state(), config=config)
    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["report_path"] == f"reports/{report_filename(_OBJECTIVE, 't-loop')}"
    assert result.get("error") is None
