"""Tests for the per-node error-boundary wrapper."""

from typing import Any

import pytest
from langgraph.errors import GraphInterrupt

from market_research_team import guardrails as guardrails_module
from market_research_team.guardrails import with_error_boundary
from market_research_team.state import AgentState


def _state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def test_with_error_boundary_passes_through_successful_result() -> None:
    def _node(_state: AgentState) -> dict[str, Any]:
        return {"research_findings": [{"source": "x", "content": "y", "relevance_score": 1.0}]}

    wrapped = with_error_boundary("research", _node)
    result = wrapped(_state())

    assert result == {
        "research_findings": [{"source": "x", "content": "y", "relevance_score": 1.0}]
    }


def test_with_error_boundary_catches_exception_and_sets_error() -> None:
    def _node(_state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    wrapped = with_error_boundary("research", _node)
    result = wrapped(_state())

    assert result["error"] == "research failed (RuntimeError): boom"
    assert result["messages"][0].name == "research"
    assert "boom" in result["messages"][0].content


def test_with_error_boundary_applies_fallback_updates_on_error() -> None:
    def _node(_state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    wrapped = with_error_boundary("supervisor", _node, fallback_updates={"next": "FINISH"})
    result = wrapped(_state())

    assert result["next"] == "FINISH"
    assert result["error"] == "supervisor failed (RuntimeError): boom"


def test_with_error_boundary_reraises_graph_interrupt_instead_of_swallowing_it() -> None:
    """A human-in-the-loop `interrupt()` call raises `GraphInterrupt`, which
    LangGraph's runtime must see -- if this bare `except Exception` caught
    it instead, every reviewer pause would look like a node crash."""

    def _node(_state: AgentState) -> dict[str, Any]:
        raise GraphInterrupt()

    wrapped = with_error_boundary("reporting", _node)

    with pytest.raises(GraphInterrupt):
        wrapped(_state())


def test_with_error_boundary_records_node_latency_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        guardrails_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    def _node(_state: AgentState) -> dict[str, Any]:
        return {}

    wrapped = with_error_boundary("research", _node)
    wrapped(_state())

    assert len(calls) == 1
    assert calls[0]["event"] == "node_latency"
    assert calls[0]["node"] == "research"
    assert calls[0]["outcome"] == "ok"
    assert calls[0]["duration_ms"] >= 0


def test_with_error_boundary_records_node_latency_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        guardrails_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    def _node(_state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    wrapped = with_error_boundary("research", _node)
    wrapped(_state())

    assert len(calls) == 1
    assert calls[0]["outcome"] == "error"
    assert calls[0]["error_type"] == "RuntimeError"
    assert calls[0]["transient"] is False


def test_with_error_boundary_records_node_latency_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        guardrails_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    def _node(_state: AgentState) -> dict[str, Any]:
        raise GraphInterrupt()

    wrapped = with_error_boundary("reporting", _node)

    with pytest.raises(GraphInterrupt):
        wrapped(_state())

    assert len(calls) == 1
    assert calls[0]["outcome"] == "interrupted"


def test_with_error_boundary_does_not_interfere_on_success_path() -> None:
    calls: list[str] = []

    def _node(state: AgentState) -> dict[str, Any]:
        calls.append(state["objective"])
        return {}

    wrapped = with_error_boundary("research", _node)
    wrapped(_state())

    assert calls == ["Assess competitor pricing strategy"]


class RateLimitError(Exception):
    """Stands in for `openai.RateLimitError` / `anthropic.RateLimitError`."""


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (TimeoutError("slow"), True),
        (ConnectionResetError("reset"), True),  # a ConnectionError subclass
        (RateLimitError("429"), True),
        (RuntimeError("bug"), False),
        (ValueError("bad input"), False),
    ],
)
def test_is_transient_classifies_by_exception_class(exc: Exception, transient: bool) -> None:
    assert guardrails_module.is_transient(exc) is transient


def test_with_error_boundary_marks_a_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        guardrails_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    def _node(_state: AgentState) -> dict[str, Any]:
        raise RateLimitError("429 too many requests")

    result = with_error_boundary("analytics", _node)(_state())

    assert result["error"] == "analytics failed (RateLimitError, transient): 429 too many requests"
    assert calls[0]["error_type"] == "RateLimitError"
    assert calls[0]["transient"] is True
