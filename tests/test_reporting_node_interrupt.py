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
from market_research_team.state import AgentState


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
    builder = StateGraph(AgentState)
    builder.add_node("reporting", reporting_node_module.reporting_node)
    builder.add_edge(START, "reporting")
    builder.add_edge("reporting", END)
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
    assert interrupt_value["filename"] == "assess-competitor-pricing-strategy.md"
    assert interrupt_value["attempt"] == 1

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["report_path"] == "reports/assess-competitor-pricing-strategy.md"
    assert result.get("error") is None
    assert "__interrupt__" not in result


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
    # LangGraph replays the node from the top on every resume, so
    # `draft_report` re-runs for every already-resolved earlier round too --
    # `feedback_log` accumulates several calls, not exactly one per round.
    # What matters is every call after the rejection carries its feedback.
    assert None in feedback_log
    assert "Add a pricing table." in feedback_log


def test_reporting_node_ends_run_after_max_review_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report_recording([]))

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t3"}}
    max_rounds = reporting_node_module._MAX_REVIEW_ROUNDS

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
