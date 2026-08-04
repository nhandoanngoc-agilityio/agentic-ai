"""Tests for the per-node error-boundary wrapper."""

from typing import Any

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

    assert result["error"] == "research failed: boom"
    assert result["messages"][0].name == "research"
    assert "boom" in result["messages"][0].content


def test_with_error_boundary_applies_fallback_updates_on_error() -> None:
    def _node(_state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    wrapped = with_error_boundary("supervisor", _node, fallback_updates={"next": "FINISH"})
    result = wrapped(_state())

    assert result["next"] == "FINISH"
    assert result["error"] == "supervisor failed: boom"


def test_with_error_boundary_does_not_interfere_on_success_path() -> None:
    calls: list[str] = []

    def _node(state: AgentState) -> dict[str, Any]:
        calls.append(state["objective"])
        return {}

    wrapped = with_error_boundary("research", _node)
    wrapped(_state())

    assert calls == ["Assess competitor pricing strategy"]
