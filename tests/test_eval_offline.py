"""Unit tests for the eval orchestration logic itself — all fake LLMs, no
real API calls. Proves the aggregation/pass-fail logic is correct;
whether real models actually pass these cases is what `scripts/run_evals.py`
is for.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from market_research_team import graph as graph_module
from market_research_team.config import settings
from market_research_team.evaluation.offline_eval import (
    evaluate_analytics,
    evaluate_full_pipeline,
    evaluate_query_rewriter,
    evaluate_reporting,
    evaluate_supervisor_decision,
    run_all,
    save_results,
)
from market_research_team.state import AgentState


class _FakeStructuredResult:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class _FakeStructuredLLM:
    def __init__(self, result: _FakeStructuredResult) -> None:
        self._result = result

    def invoke(self, _messages: list[object]) -> _FakeStructuredResult:
        return self._result


class _FakeQueryRewriteLLM:
    """Satisfies rewrite_and_expand's `.with_structured_output(...)` usage."""

    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    def with_structured_output(self, _schema: object) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(_FakeStructuredResult(queries=self._queries))


class _FakeSupervisorLLM:
    """Satisfies decide_next_step's `.with_structured_output(...)` usage."""

    def __init__(self, next_step: str) -> None:
        self._next_step = next_step

    def with_structured_output(self, _schema: object) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(_FakeStructuredResult(next=self._next_step))


class _FakeToolBoundLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages: list[object]) -> AIMessage:
        return self._responses.pop(0)


class _FakeAnalyticsLLM:
    """Satisfies run_tool_calling_loop's `.bind_tools(...)` usage."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses

    def bind_tools(self, _tools: list[object]) -> _FakeToolBoundLLM:
        return _FakeToolBoundLLM(self._responses)


class _FakeReportingLLM:
    """Satisfies draft_report's plain `.invoke(...)` usage."""

    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages: list[object]) -> AIMessage:
        return AIMessage(content=self._content)


def test_evaluate_query_rewriter_passes_on_good_queries() -> None:
    llm = _FakeQueryRewriteLLM(["acme pricing", "globex pricing"])

    results = evaluate_query_rewriter(llm, "fake-provider")  # type: ignore[arg-type]

    assert results
    assert all(result.passed for result in results if result.case_name == "acme_vs_globex_pricing")


def test_evaluate_query_rewriter_fails_without_keyword_coverage() -> None:
    llm = _FakeQueryRewriteLLM(["unrelated topic entirely"])

    results = evaluate_query_rewriter(llm, "fake-provider")  # type: ignore[arg-type]

    acme_result = next(r for r in results if r.case_name == "acme_vs_globex_pricing")
    assert not acme_result.passed


def test_evaluate_supervisor_decision_passes_on_allowed_choice() -> None:
    llm = _FakeSupervisorLLM("analytics")

    results = evaluate_supervisor_decision(llm, "fake-provider")  # type: ignore[arg-type]

    assert all(result.passed for result in results)


def test_evaluate_analytics_passes_when_grounded_value_computed() -> None:
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "value_range",
                "args": {"values": [150000, 400000]},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final_message = AIMessage(content="Done.", tool_calls=[])
    llm = _FakeAnalyticsLLM([tool_call_message, final_message])

    results = evaluate_analytics(llm, "fake-provider")  # type: ignore[arg-type]

    assert results[0].passed


def test_evaluate_analytics_fails_when_no_tool_called() -> None:
    final_message = AIMessage(content="I don't know.", tool_calls=[])
    llm = _FakeAnalyticsLLM([final_message])

    results = evaluate_analytics(llm, "fake-provider")  # type: ignore[arg-type]

    assert not results[0].passed


def test_evaluate_reporting_passes_when_facts_and_sections_present() -> None:
    llm = _FakeReportingLLM("# Objective\nSummary.\n# Findings\nAcme is $49.\n# Analysis\nDone.")

    results = evaluate_reporting(llm, "fake-provider")  # type: ignore[arg-type]

    assert results[0].passed


def test_evaluate_reporting_fails_when_a_required_fact_is_missing() -> None:
    content = "# Objective\nSummary.\n# Findings\nNo numbers here.\n# Analysis\nDone."
    llm = _FakeReportingLLM(content)

    results = evaluate_reporting(llm, "fake-provider")  # type: ignore[arg-type]

    assert not results[0].passed


def test_evaluate_full_pipeline_passes_on_clean_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_graph(state: AgentState, **_kwargs: object) -> AgentState:
        return {
            **state,
            "research_findings": [{"source": "x", "content": "y", "relevance_score": 0.9}],
            "analytics_results": [{"metric": "m", "value": 1.0, "detail": "d"}],
            "report_path": "reports/mock.md",
            "error": None,
        }

    monkeypatch.setattr(graph_module, "run_graph", _fake_run_graph)

    results = evaluate_full_pipeline("fake-provider")

    assert results[0].passed


def test_evaluate_full_pipeline_fails_when_error_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_graph(state: AgentState, **_kwargs: object) -> AgentState:
        return {**state, "error": "something failed"}

    monkeypatch.setattr(graph_module, "run_graph", _fake_run_graph)

    results = evaluate_full_pipeline("fake-provider")

    assert not results[0].passed


def test_run_all_restores_llm_provider_even_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("market_research_team.evaluation.offline_eval.get_chat_model", _boom)

    with pytest.raises(RuntimeError):
        run_all("openai")

    assert settings.llm_provider == "anthropic"


def test_save_results_writes_a_timestamped_json_file(tmp_path: Path) -> None:
    from market_research_team.evaluation.offline_eval import EvalResult

    results = [EvalResult("category", "case", True, "detail", "provider")]

    path = save_results(results, tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == [
        {
            "category": "category",
            "case_name": "case",
            "passed": True,
            "detail": "detail",
            "provider": "provider",
        }
    ]
