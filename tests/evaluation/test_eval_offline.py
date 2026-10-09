"""Unit tests for the eval orchestration logic itself — all fake LLMs, no
real API calls. Proves the aggregation/pass-fail logic is correct;
whether real models actually pass these cases is what `scripts/run_evals.py`
is for.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from market_research_team import graph as graph_module
from market_research_team.config import settings
from market_research_team.evaluation import offline_eval
from market_research_team.evaluation.golden_dataset import FULL_PIPELINE_CASES
from market_research_team.evaluation.offline_eval import (
    evaluate_analytics,
    evaluate_full_pipeline,
    evaluate_query_rewriter,
    evaluate_reporting,
    evaluate_retrieval,
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

    def invoke(self, _messages: list[object], config: object = None) -> _FakeStructuredResult:
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

    def invoke(self, _messages: list[object], config: object = None) -> AIMessage:
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

    def invoke(self, _messages: list[object], config: object = None) -> AIMessage:
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


def test_evaluate_retrieval_passes_when_expected_source_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_research_pipeline(objective: str):
        findings = [{"source": "competitor_acme.md", "content": "x", "relevance_score": 0.9}]
        return findings, 1, 1, []

    monkeypatch.setattr(offline_eval, "run_research_pipeline", _fake_run_research_pipeline)

    results = evaluate_retrieval("fake-provider")

    assert results
    assert all(result.passed for result in results if result.case_name == "acme_pricing_retrieval")


def test_evaluate_retrieval_fails_when_expected_source_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_research_pipeline(objective: str):
        return [], 1, 0, []

    monkeypatch.setattr(offline_eval, "run_research_pipeline", _fake_run_research_pipeline)

    results = evaluate_retrieval("fake-provider")

    assert all(not result.passed for result in results)


class _Coverage:
    def __init__(self, item_id: str, status: str, *sources: str) -> None:
        self.id, self.status, self.sources = item_id, status, list(sources)


def _supervisor_answering(next_step: str, focus_id: str | None = None) -> object:
    """A supervisor model that judges the pricing plan correctly (Acme from
    competitor_acme.md, Globex from competitor_globex.md when present) and
    answers `next_step`."""

    class _LLM:
        def with_structured_output(self, _schema: object) -> object:
            return self

        def invoke(self, messages: list[Any], config: object = None) -> _FakeStructuredResult:
            context = messages[1].content
            coverage = [_Coverage("q1", "answered", "competitor_acme.md")]
            if "competitor_globex.md" in context:
                coverage.append(_Coverage("q2", "answered", "competitor_globex.md"))
            return _FakeStructuredResult(
                next=next_step, focus_id=focus_id, rationale="r", coverage=coverage
            )

    return _LLM()


def _supervisor_results(llm: object) -> dict[str, bool]:
    results = evaluate_supervisor_decision(llm, "fake-provider")  # type: ignore[arg-type]
    return {result.case_name: result.passed for result in results}


def test_supervisor_cases_have_one_right_answer() -> None:
    """Before, every allowed choice passed, so no supervisor case could fail."""

    # On the covered plan a "research" answer is overridden by rule (nothing
    # left to research) and reaches reporting, so both cases pass.
    # In the teaser case q2 is the market-size gap, so researching it is right.
    assert _supervisor_results(_supervisor_answering("research", "q2")) == {
        "acme_covered_globex_missing": True,
        "plan_covered_analysis_done": True,
        "teaser_does_not_answer_market_size": True,
    }
    assert _supervisor_results(_supervisor_answering("reporting")) == {
        "acme_covered_globex_missing": False,
        "plan_covered_analysis_done": True,
        "teaser_does_not_answer_market_size": False,
    }
    assert not any(_supervisor_results(_FakeSupervisorLLM("analytics")).values())


def test_a_hand_back_must_target_the_gap_not_just_choose_research() -> None:
    """Without judging q1 answered, the hand-back defaults to q1 (Acme), which
    is not the gap: the case fails on its focus."""

    results = evaluate_supervisor_decision(_FakeSupervisorLLM("research"), "fake-provider")  # type: ignore[arg-type]

    (gap_case,) = [r for r in results if r.case_name == "acme_covered_globex_missing"]
    assert not gap_case.passed
    assert "focus missing ['globex']" in gap_case.detail


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
    submit = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "submit_analysis",
                "args": {"insights": [{"text": "Globex deals span $250K.", "result_ids": ["r1"]}]},
                "id": "call-2",
                "type": "tool_call",
            }
        ],
    )
    llm = _FakeAnalyticsLLM([tool_call_message, submit])

    results = evaluate_analytics(llm, "fake-provider")  # type: ignore[arg-type]

    assert results[0].passed, results[0].detail


def test_evaluate_analytics_fails_without_a_submitted_insight() -> None:
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
    llm = _FakeAnalyticsLLM([tool_call_message, AIMessage(content="Done.")])

    results = evaluate_analytics(llm, "fake-provider")  # type: ignore[arg-type]

    assert not results[0].passed
    assert "0 insight(s)" in results[0].detail


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


def test_save_results_writes_results_json_in_the_run_dir(tmp_path: Path) -> None:
    from market_research_team import versioning
    from market_research_team.evaluation.offline_eval import EvalResult

    results = [EvalResult("category", "case", True, "detail", "provider", 0.5, 1)]

    path = save_results(results, tmp_path / "eval_1", gate=[{"passed": True}])

    assert path == tmp_path / "eval_1" / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["results"][0]["judge_score"] == 0.5 and data["results"][0]["repeat"] == 1
    assert data["gate"] == [{"passed": True}]
    assert data["agent_version"] == versioning.agent_version()
    assert data["manifest"]["fingerprint"] == versioning.build_manifest()["fingerprint"]


def test_scoped_eval_run_redirects_and_restores_paths(tmp_path: Path) -> None:
    from market_research_team.evaluation.offline_eval import scoped_eval_run

    original_audit, original_reports = settings.audit_log_path, settings.reports_dir

    with scoped_eval_run(tmp_path / "run") as run_dir:
        assert settings.audit_log_path == run_dir / "audit.jsonl"
        assert settings.reports_dir == run_dir / "reports"

    assert settings.audit_log_path == original_audit
    assert settings.reports_dir == original_reports


def test_scoped_eval_run_restores_paths_on_error(tmp_path: Path) -> None:
    from market_research_team.evaluation.offline_eval import scoped_eval_run

    original_audit = settings.audit_log_path

    with pytest.raises(RuntimeError):
        with scoped_eval_run(tmp_path / "run"):
            raise RuntimeError("boom")

    assert settings.audit_log_path == original_audit


def test_run_all_includes_safety_and_passes_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def record(name):
        def _fn(*args, **kwargs):
            calls[name] = kwargs
            return []

        return _fn

    monkeypatch.setattr(offline_eval, "get_chat_model", lambda: object())
    for name in (
        "evaluate_query_rewriter",
        "evaluate_planner",
        "evaluate_retrieval",
        "evaluate_supervisor_decision",
        "evaluate_analytics",
        "evaluate_reporting",
        "evaluate_full_pipeline",
        "evaluate_safety",
        "evaluate_regressions",
    ):
        monkeypatch.setattr(offline_eval, name, record(name))

    offline_eval.run_all("anthropic", repeat=2, judge_llm="J")  # type: ignore[arg-type]

    assert calls["evaluate_safety"] == {"repeat": 2}
    assert calls["evaluate_reporting"] == {"judge_llm": "J", "repeat": 2}
    assert calls["evaluate_retrieval"] == {"repeat": 2}
    assert calls["evaluate_regressions"] == {"repeat": 2}
    assert calls["evaluate_planner"] == {"repeat": 2}


class _FakeJudge:
    def __init__(self, score: float | None = None, error: Exception | None = None) -> None:
        self._score = score
        self._error = error

    def with_structured_output(self, _schema: object) -> "_FakeJudge":
        return self

    def invoke(self, _messages: list[object]) -> _FakeStructuredResult:
        if self._error:
            raise self._error
        return _FakeStructuredResult(passed=True, score=self._score, reasoning="r")


def test_evaluators_record_judge_score_and_repeat() -> None:
    llm = _FakeQueryRewriteLLM(["acme pricing", "globex pricing"])

    results = evaluate_query_rewriter(
        llm,  # type: ignore[arg-type]
        "p",
        judge_llm=_FakeJudge(0.75),  # type: ignore[arg-type]
        repeat=2,
    )

    assert all(r.judge_score == 0.75 and r.repeat == 2 for r in results)


def test_judge_exception_leaves_case_unjudged() -> None:
    llm = _FakeQueryRewriteLLM(["acme pricing"])

    results = evaluate_query_rewriter(
        llm,  # type: ignore[arg-type]
        "p",
        judge_llm=_FakeJudge(error=RuntimeError("bad json")),  # type: ignore[arg-type]
    )

    assert all(r.judge_score is None for r in results)


def test_judge_score_is_clamped_to_unit_interval() -> None:
    llm = _FakeQueryRewriteLLM(["acme pricing"])

    results = evaluate_query_rewriter(llm, "p", judge_llm=_FakeJudge(7.0))  # type: ignore[arg-type]

    assert all(r.judge_score == 1.0 for r in results)


def test_evaluate_analytics_emits_tool_selection_result() -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "value_range", "args": {"values": [150000, 400000]}, "id": "1"}],
    )
    llm = _FakeAnalyticsLLM([tool_call, AIMessage(content="done")])

    results = evaluate_analytics(llm, "p")  # type: ignore[arg-type]

    selection = [r for r in results if r.category == "tool_selection"]
    assert selection and selection[0].passed
    assert "value_range" in selection[0].detail


def test_evaluate_full_pipeline_answers_approval_with_a_thread_id(monkeypatch) -> None:
    from langgraph.types import Command

    seen: dict[str, object] = {}

    def fake_run_graph(state, *, compiled_graph=None, thread_id=None, **_):
        seen["thread_id"] = thread_id
        if isinstance(state, Command):
            seen["resume"] = state.resume
            return {"report_path": "reports/mock.md", "error": None}
        return {"__interrupt__": ["pending"]}

    monkeypatch.setattr(graph_module, "run_graph", fake_run_graph)

    results = evaluate_full_pipeline("p", repeat=1)

    assert results[0].passed
    assert seen["resume"] == {"approved": True}
    assert str(seen["thread_id"]).startswith("eval-") and str(seen["thread_id"]).endswith("-r1")


def test_judge_error_is_recorded_in_the_detail() -> None:
    llm = _FakeQueryRewriteLLM(["acme pricing"])

    results = evaluate_query_rewriter(
        llm,  # type: ignore[arg-type]
        "p",
        judge_llm=_FakeJudge(error=RuntimeError("no API key for judge")),  # type: ignore[arg-type]
    )

    assert all("judge_error=RuntimeError: no API key for judge" in r.detail for r in results)


class _FakePlanLLM:
    def __init__(self, questions: list[str]) -> None:
        self._questions = questions

    def with_structured_output(self, _schema: object) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(_FakeStructuredResult(questions=self._questions))


def test_evaluate_planner_passes_a_per_company_plan_and_fails_a_restated_one() -> None:
    from market_research_team.evaluation.offline_eval import evaluate_planner

    per_company = _FakePlanLLM(["What does Acme charge per seat?", "What is Globex's ACV?"])
    good = evaluate_planner(per_company, "p")  # type: ignore[arg-type]
    restated = evaluate_planner(
        _FakePlanLLM(["How do Acme and Globex prices compare?", "Who are Acme's customers?"]),  # type: ignore[arg-type]
        "p",
    )

    assert [r.category for r in good] == ["planning"]
    assert good[0].passed
    assert not restated[0].passed


def test_reporting_case_with_feedback_requires_a_table() -> None:
    text = "# Objective\nx\n# Findings\nAcme $49, Globex $150K.\n# Analysis\ny"

    without = {r.case_name: r for r in evaluate_reporting(_FakeReportingLLM(text), "p")}  # type: ignore[arg-type]
    table = text + "\n\n| Vendor | Price |\n|---|---|\n| Acme | $49 |"
    with_table = {r.case_name: r for r in evaluate_reporting(_FakeReportingLLM(table), "p")}  # type: ignore[arg-type]

    assert not without["reviewer_feedback_adds_table"].passed
    assert "no markdown table" in without["reviewer_feedback_adds_table"].detail
    assert with_table["reviewer_feedback_adds_table"].passed


def test_evaluate_full_pipeline_adds_a_trajectory_result_from_the_audit_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from market_research_team.security import audit

    def _fake_run_graph(state: AgentState, *, thread_id: str, **_kwargs: object) -> AgentState:
        route = [("research", "rule"), ("reporting", "llm"), ("FINISH", "rule")]
        for next_step, decided_by in route:
            audit.record("route_decision", thread_id, next=next_step, decided_by=decided_by)
        audit.record("route_decision", "another-thread", next="research", decided_by="fallback")
        return {
            **state,
            "plan": [{"id": "q1", "question": "Q?", "status": "answered", "sources": ["x"]}],
            "report_path": "reports/mock.md",
            "error": None,
        }

    monkeypatch.setattr(graph_module, "run_graph", _fake_run_graph)

    results = evaluate_full_pipeline("fake-provider")

    trajectories = [r for r in results if r.category == "trajectory"]
    assert len(trajectories) == len(FULL_PIPELINE_CASES)
    for trajectory in trajectories:
        assert trajectory.passed, trajectory.detail
        assert "research(rule) -> reporting(llm) -> FINISH(rule)" in trajectory.detail


def test_conflict_reporting_cases_fail_an_unlabelled_old_figure() -> None:
    body = "Acme charges $55. Globex runs $150K to $400K."
    labelled = f"# Objective\nx\n# Findings\n{body} Formerly $49, $120K to $350K.\n# Analysis\ny"
    unlabelled = f"# Objective\nx\n# Findings\n{body} Also $49, $120K to $350K.\n# Analysis\ny"

    good = {r.case_name: r for r in evaluate_reporting(_FakeReportingLLM(labelled), "p")}  # type: ignore[arg-type]
    bad = {r.case_name: r for r in evaluate_reporting(_FakeReportingLLM(unlabelled), "p")}  # type: ignore[arg-type]

    for name in ("acme_price_conflict", "globex_acv_conflict"):
        assert good[name].passed, good[name].detail
        assert not bad[name].passed
        assert "unlabelled outdated figures" in bad[name].detail


def test_golden_dataset_has_the_freshness_cases() -> None:
    from market_research_team.evaluation import golden_dataset as gd

    names = {
        case.name
        for cases in (
            gd.RETRIEVAL_CASES,
            gd.REPORTING_CASES,
            gd.ANALYTICS_CASES,
            gd.SUPERVISOR_DECISION_CASES,
            gd.FULL_PIPELINE_CASES,
        )
        for case in cases
    }

    assert {
        "initech_pricing_retrieval",
        "embedded_analytics_trends_retrieval",
        "acme_price_conflict",
        "globex_acv_conflict",
        "globex_acv_midpoint_uses_newest",
        "teaser_does_not_answer_market_size",
        "three_vendor_pricing",
    } <= names
