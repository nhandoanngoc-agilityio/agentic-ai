"""Tests for `graph._apply_response_cache`: the read side of the opt-in
full-pipeline response cache (see caching/response_cache.py)."""

import pytest
from langgraph.types import Command

from market_research_team import graph as graph_module
from market_research_team.config import settings
from market_research_team.state import AgentState

_FINDINGS = [{"source": "competitor_acme.md", "content": "x", "relevance_score": 0.9}]
_RESULTS = [{"metric": "mean", "value": 1.0, "detail": "d", "entity": None}]


def _initial_state(objective: str = "Assess competitor pricing strategy") -> AgentState:
    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def test_disabled_returns_state_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "response_cache_enabled", False)

    state = _initial_state()
    result = graph_module._apply_response_cache(state)

    assert result is state


@pytest.mark.parametrize("resume_input", [Command(resume={"approved": True}), None])
def test_a_resume_or_retry_never_consults_the_cache(
    monkeypatch: pytest.MonkeyPatch, resume_input: object
) -> None:
    def _must_not_be_called(_state: object) -> None:
        raise AssertionError("the response cache is for fresh runs only")

    seen: list[object] = []
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    monkeypatch.setattr(graph_module, "_apply_response_cache", _must_not_be_called)
    monkeypatch.setattr(
        graph_module, "_execute", lambda graph, state, config, on_step: seen.append(state) or {}
    )

    graph_module.run_graph(resume_input, compiled_graph=object())  # type: ignore[arg-type]

    assert seen == [resume_input]


def test_cache_hit_populates_findings_and_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    monkeypatch.setattr(
        graph_module, "get_cached_response", lambda objective, persist_dir: (_FINDINGS, _RESULTS)
    )

    result = graph_module._apply_response_cache(_initial_state())

    assert result["research_findings"] == _FINDINGS
    assert result["analytics_results"] == _RESULTS
    assert result["from_response_cache"] is True


def test_cache_miss_leaves_state_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    monkeypatch.setattr(graph_module, "get_cached_response", lambda objective, persist_dir: None)

    state = _initial_state()
    result = graph_module._apply_response_cache(state)

    assert result == state
    assert "from_response_cache" not in result


def test_invalid_objective_is_left_for_input_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "response_cache_enabled", True)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("objective must not be empty")

    monkeypatch.setattr(graph_module, "validate_objective", _boom)

    state = _initial_state(objective="")
    result = graph_module._apply_response_cache(state)

    assert result is state
