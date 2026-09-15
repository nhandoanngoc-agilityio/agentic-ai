"""Hermetic tests for the LangSmith eval integration -- no network, no real
LangSmith calls. A fake Client double stands in for langsmith.Client so
sync_dataset's get-or-create-and-wipe logic is exercised deterministically.
"""

import pytest

from market_research_team.evaluation import langsmith_eval


class _FakeDataset:
    def __init__(self, name: str) -> None:
        self.id = f"dataset-{name}"
        self.name = name


class _FakeExampleRecord:
    def __init__(self, example_id: str) -> None:
        self.id = example_id


class _FakeLangSmithClient:
    """Hermetic stand-in for langsmith.Client -- records calls instead of hitting the network."""

    def __init__(
        self, *, existing_dataset: bool = False, existing_example_ids: list[str] | None = None
    ) -> None:
        self._existing_dataset = existing_dataset
        self._example_ids = list(existing_example_ids or [])
        self.deleted_example_ids: list[str] = []
        self.created_examples: list[dict] = []
        self.created_dataset_names: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self._existing_dataset

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> _FakeDataset:
        self.created_dataset_names.append(dataset_name)
        return _FakeDataset(dataset_name)

    def read_dataset(self, *, dataset_name: str) -> _FakeDataset:
        return _FakeDataset(dataset_name)

    def list_examples(self, *, dataset_name: str) -> list[_FakeExampleRecord]:
        return [_FakeExampleRecord(example_id) for example_id in self._example_ids]

    def delete_example(self, *, example_id: str) -> None:
        self.deleted_example_ids.append(example_id)

    def create_example(self, *, inputs: dict, outputs: dict, dataset_id: str) -> None:
        self.created_examples.append(
            {"inputs": inputs, "outputs": outputs, "dataset_id": dataset_id}
        )


def test_sync_dataset_creates_new_dataset_when_absent() -> None:
    client = _FakeLangSmithClient(existing_dataset=False)
    examples = [{"inputs": {"objective": "x"}, "outputs": {"min_queries": 1}}]

    dataset = langsmith_eval.sync_dataset(client, "query_rewrite", examples)  # type: ignore[arg-type]

    assert client.created_dataset_names == ["market-research-team-query_rewrite"]
    assert client.created_examples == [
        {"inputs": {"objective": "x"}, "outputs": {"min_queries": 1}, "dataset_id": dataset.id}
    ]
    assert client.deleted_example_ids == []


def test_sync_dataset_wipes_existing_examples_before_recreating() -> None:
    client = _FakeLangSmithClient(existing_dataset=True, existing_example_ids=["ex-1", "ex-2"])
    examples = [{"inputs": {}, "outputs": {}}]

    langsmith_eval.sync_dataset(client, "analytics", examples)  # type: ignore[arg-type]

    assert client.deleted_example_ids == ["ex-1", "ex-2"]
    assert client.created_dataset_names == []
    assert len(client.created_examples) == 1


class _FakeRun:
    def __init__(self, outputs: dict | None) -> None:
        self.outputs = outputs


class _FakeExample:
    def __init__(self, inputs: dict | None = None, outputs: dict | None = None) -> None:
        self.inputs = inputs
        self.outputs = outputs


class _FakeStructuredJudgeLLM:
    """Satisfies `llm.with_structured_output(_JudgeScoreSchema).invoke(...)`."""

    def __init__(self, score: float, passed: bool, reasoning: str = "because") -> None:
        self._result = langsmith_eval._JudgeScoreSchema(
            passed=passed, score=score, reasoning=reasoning
        )

    def with_structured_output(self, _schema: object) -> "_FakeStructuredJudgeLLM":
        return self

    def invoke(self, _messages: list[object]) -> langsmith_eval._JudgeScoreSchema:
        return self._result


def test_query_rewrite_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import QUERY_REWRITE_CASES

    examples = langsmith_eval._query_rewrite_examples()

    assert len(examples) == len(QUERY_REWRITE_CASES)
    assert examples[0]["inputs"] == {"objective": QUERY_REWRITE_CASES[0].objective}
    assert examples[0]["outputs"]["min_queries"] == QUERY_REWRITE_CASES[0].min_queries


def test_target_query_rewrite_calls_rewrite_and_expand(monkeypatch) -> None:
    monkeypatch.setattr(
        langsmith_eval, "rewrite_and_expand", lambda objective, llm: [f"{objective}-q1"]
    )

    result = langsmith_eval.target_query_rewrite({"objective": "acme pricing"}, llm=object())

    assert result == {"queries": ["acme pricing-q1"]}


def test_deterministic_evaluator_query_rewrite_passes_on_good_output() -> None:
    run = _FakeRun(outputs={"queries": ["acme pricing", "globex pricing"]})
    example = _FakeExample(
        outputs={
            "min_queries": 1,
            "max_queries": 4,
            "required_any_keywords": ["pricing"],
        }
    )

    result = langsmith_eval.deterministic_evaluator_query_rewrite(run, example)  # type: ignore[arg-type]

    assert result["key"] == "deterministic"
    assert result["score"] == 1.0


def test_deterministic_evaluator_query_rewrite_fails_without_keyword_coverage() -> None:
    run = _FakeRun(outputs={"queries": ["unrelated topic"]})
    example = _FakeExample(
        outputs={
            "min_queries": 1,
            "max_queries": 4,
            "required_any_keywords": ["pricing"],
        }
    )

    result = langsmith_eval.deterministic_evaluator_query_rewrite(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_query_rewrite_returns_judge_score() -> None:
    run = _FakeRun(outputs={"queries": ["acme pricing"]})
    example = _FakeExample(inputs={"objective": "Compare Acme and Globex pricing"})
    fake_llm = _FakeStructuredJudgeLLM(score=0.9, passed=True, reasoning="good coverage")

    result = langsmith_eval.llm_judge_query_rewrite(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.9, "comment": "good coverage"}


def test_supervisor_decision_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import SUPERVISOR_DECISION_CASES

    examples = langsmith_eval._supervisor_decision_examples()

    assert len(examples) == len(SUPERVISOR_DECISION_CASES)
    assert examples[0]["outputs"]["allowed_decisions"] == list(
        SUPERVISOR_DECISION_CASES[0].allowed_decisions
    )


def test_target_supervisor_decision_calls_decide_next_step(monkeypatch) -> None:
    monkeypatch.setattr(langsmith_eval, "decide_next_step", lambda state, llm: "analytics")

    result = langsmith_eval.target_supervisor_decision(
        {"research_findings": [], "analytics_results": [], "report_path": None}, llm=object()
    )

    assert result == {"decision": "analytics"}


def test_deterministic_evaluator_supervisor_decision_passes_on_allowed_choice() -> None:
    run = _FakeRun(outputs={"decision": "analytics"})
    example = _FakeExample(outputs={"allowed_decisions": ["research", "analytics"]})

    result = langsmith_eval.deterministic_evaluator_supervisor_decision(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_supervisor_decision_fails_on_disallowed_choice() -> None:
    run = _FakeRun(outputs={"decision": "reporting"})
    example = _FakeExample(outputs={"allowed_decisions": ["research", "analytics"]})

    result = langsmith_eval.deterministic_evaluator_supervisor_decision(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_supervisor_decision_returns_judge_score() -> None:
    run = _FakeRun(outputs={"decision": "analytics"})
    example = _FakeExample(inputs={"research_findings": [{"source": "x"}], "analytics_results": []})
    fake_llm = _FakeStructuredJudgeLLM(score=0.8, passed=True, reasoning="reasonable")

    result = langsmith_eval.llm_judge_supervisor_decision(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.8, "comment": "reasonable"}


def test_analytics_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import ANALYTICS_CASES

    examples = langsmith_eval._analytics_examples()

    assert len(examples) == len(ANALYTICS_CASES)
    assert examples[0]["outputs"]["plausible_values"] == ANALYTICS_CASES[0].plausible_values


def test_target_analytics_calls_run_tool_calling_loop(monkeypatch) -> None:
    fake_results = [{"metric": "mean", "value": 275000.0, "detail": "d"}]
    monkeypatch.setattr(
        langsmith_eval,
        "run_tool_calling_loop",
        lambda llm, tools, objective, findings: fake_results,
    )

    result = langsmith_eval.target_analytics({"objective": "x", "findings": []}, llm=object())

    assert result == {"results": fake_results}


def test_deterministic_evaluator_analytics_passes_on_grounded_value() -> None:
    run = _FakeRun(outputs={"results": [{"metric": "mean", "value": 275000.0, "detail": "d"}]})
    example = _FakeExample(
        outputs={
            "min_tool_calls": 1,
            "plausible_values": [275000.0],
            "tolerance": 1.0,
        }
    )

    result = langsmith_eval.deterministic_evaluator_analytics(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_analytics_fails_when_no_results() -> None:
    run = _FakeRun(outputs={"results": []})
    example = _FakeExample(
        outputs={
            "min_tool_calls": 1,
            "plausible_values": [275000.0],
            "tolerance": 1.0,
        }
    )

    result = langsmith_eval.deterministic_evaluator_analytics(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_analytics_returns_judge_score() -> None:
    run = _FakeRun(outputs={"results": [{"metric": "mean", "value": 275000.0, "detail": "d"}]})
    example = _FakeExample(inputs={"findings": [{"source": "x", "content": "y"}]})
    fake_llm = _FakeStructuredJudgeLLM(score=1.0, passed=True, reasoning="grounded")

    result = langsmith_eval.llm_judge_analytics(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 1.0, "comment": "grounded"}


def test_reporting_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import REPORTING_CASES

    examples = langsmith_eval._reporting_examples()

    assert len(examples) == len(REPORTING_CASES)
    assert examples[0]["outputs"]["required_facts"] == REPORTING_CASES[0].required_facts


def test_target_reporting_calls_draft_report(monkeypatch) -> None:
    monkeypatch.setattr(
        langsmith_eval, "draft_report", lambda objective, findings, results, llm: "# Report body"
    )

    result = langsmith_eval.target_reporting(
        {"objective": "x", "findings": [], "results": []}, llm=object()
    )

    assert result == {"report": "# Report body"}


def test_deterministic_evaluator_reporting_passes_when_facts_and_sections_present() -> None:
    report = "# Objective\nSummary.\n# Findings\nAcme is $49.\n# Analysis\nDone."
    run = _FakeRun(outputs={"report": report})
    example = _FakeExample(
        outputs={"required_sections": ["Objective", "Findings"], "required_facts": ["49"]}
    )

    result = langsmith_eval.deterministic_evaluator_reporting(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_reporting_fails_when_a_fact_is_missing() -> None:
    report = "# Objective\nSummary.\n# Findings\nNo numbers here."
    run = _FakeRun(outputs={"report": report})
    example = _FakeExample(
        outputs={"required_sections": ["Objective"], "required_facts": ["49"]}
    )

    result = langsmith_eval.deterministic_evaluator_reporting(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_reporting_returns_judge_score() -> None:
    run = _FakeRun(outputs={"report": "# Objective\nSummary."})
    example = _FakeExample(inputs={"findings": [], "results": []})
    fake_llm = _FakeStructuredJudgeLLM(score=0.7, passed=True, reasoning="faithful")

    result = langsmith_eval.llm_judge_reporting(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.7, "comment": "faithful"}


def test_full_pipeline_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import FULL_PIPELINE_CASES

    examples = langsmith_eval._full_pipeline_examples()

    assert len(examples) == len(FULL_PIPELINE_CASES)
    assert examples[0]["inputs"] == {"objective": FULL_PIPELINE_CASES[0].objective}
    assert examples[0]["outputs"] == {}


def test_target_full_pipeline_calls_run_graph(monkeypatch) -> None:
    def _fake_run_graph(state, **_kwargs):
        return {
            **state,
            "research_findings": [{"source": "x", "content": "y", "relevance_score": 0.9}],
            "analytics_results": [],
            "report_path": "reports/mock.md",
            "error": None,
        }

    monkeypatch.setattr(langsmith_eval.graph_module, "run_graph", _fake_run_graph)

    result = langsmith_eval.target_full_pipeline({"objective": "assess pricing"}, llm=object())

    assert result == {
        "error": None,
        "report_path": "reports/mock.md",
        "findings_count": 1,
        "results_count": 0,
    }


def test_deterministic_evaluator_full_pipeline_passes_on_clean_run() -> None:
    run = _FakeRun(outputs={"error": None, "report_path": "reports/mock.md"})
    example = _FakeExample(inputs={"objective": "x"})

    result = langsmith_eval.deterministic_evaluator_full_pipeline(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_full_pipeline_fails_when_error_is_set() -> None:
    run = _FakeRun(outputs={"error": "boom", "report_path": None})
    example = _FakeExample(inputs={"objective": "x"})

    result = langsmith_eval.deterministic_evaluator_full_pipeline(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_full_pipeline_reads_report_file_and_returns_judge_score(tmp_path) -> None:
    report_file = tmp_path / "mock.md"
    report_file.write_text("# Objective\nAcme vs Globex pricing.\n", encoding="utf-8")
    run = _FakeRun(outputs={"report_path": str(report_file)})
    example = _FakeExample(inputs={"objective": "Assess Acme vs Globex pricing"})
    fake_llm = _FakeStructuredJudgeLLM(score=0.95, passed=True, reasoning="on target")

    result = langsmith_eval.llm_judge_full_pipeline(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.95, "comment": "on target"}


def test_run_langsmith_eval_computes_pass_rate_and_restores_provider(monkeypatch) -> None:
    from market_research_team.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(langsmith_eval, "get_chat_model", lambda: object())
    monkeypatch.setattr(langsmith_eval, "Client", lambda: _FakeLangSmithClient())
    monkeypatch.setattr(
        langsmith_eval, "sync_dataset", lambda client, category, examples: _FakeDataset(category)
    )

    class _FakeEvalResult:
        def __init__(self, score: float) -> None:
            self.score = score

    class _FakeExperimentResults:
        def __init__(self, name: str, rows: list[dict]) -> None:
            self._name = name
            self._rows = rows

        def experiment_name(self) -> str:
            return self._name

        def __iter__(self):
            return iter(self._rows)

    def _fake_evaluate(_target, *, data, evaluators, experiment_prefix, client):
        # data/evaluators/client are unused here but must keep these exact names --
        # run_langsmith_eval calls evaluate(..., data=..., evaluators=..., client=...) by keyword.
        return _FakeExperimentResults(
            f"{experiment_prefix}-1",
            [{"evaluation_results": {"results": [_FakeEvalResult(1.0), _FakeEvalResult(1.0)]}}],
        )

    monkeypatch.setattr(langsmith_eval, "evaluate", _fake_evaluate)

    summaries = langsmith_eval.run_langsmith_eval("openai")

    assert len(summaries) == 5
    assert {summary.category for summary in summaries} == {
        "query_rewrite",
        "supervisor_decision",
        "analytics",
        "reporting",
        "full_pipeline",
    }
    assert all(summary.pass_rate == 1.0 for summary in summaries)
    assert settings.llm_provider == "anthropic"


def test_run_langsmith_eval_computes_partial_pass_rate(monkeypatch) -> None:
    from market_research_team.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(langsmith_eval, "get_chat_model", lambda: object())
    monkeypatch.setattr(langsmith_eval, "Client", lambda: _FakeLangSmithClient())
    monkeypatch.setattr(
        langsmith_eval, "sync_dataset", lambda client, category, examples: _FakeDataset(category)
    )

    class _FakeEvalResult:
        def __init__(self, score: float) -> None:
            self.score = score

    class _FakeExperimentResults:
        def __init__(self, name: str, rows: list[dict]) -> None:
            self._name = name
            self._rows = rows

        def experiment_name(self) -> str:
            return self._name

        def __iter__(self):
            return iter(self._rows)

    def _fake_evaluate(_target, *, data, evaluators, experiment_prefix, client):
        # 3 rows: 2 fully passing (both evaluators score 1.0), 1 failing (one evaluator
        # scores 0.0) -- pass_rate should be 2/3, not 1.0 and not 0.0.
        return _FakeExperimentResults(
            f"{experiment_prefix}-1",
            [
                {"evaluation_results": {"results": [_FakeEvalResult(1.0), _FakeEvalResult(1.0)]}},
                {"evaluation_results": {"results": [_FakeEvalResult(1.0), _FakeEvalResult(1.0)]}},
                {"evaluation_results": {"results": [_FakeEvalResult(0.0), _FakeEvalResult(1.0)]}},
            ],
        )

    monkeypatch.setattr(langsmith_eval, "evaluate", _fake_evaluate)

    summaries = langsmith_eval.run_langsmith_eval("openai")

    assert len(summaries) == 5
    assert all(summary.pass_rate == 2 / 3 for summary in summaries)


def test_run_langsmith_eval_restores_provider_even_on_failure(monkeypatch) -> None:
    from market_research_team.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(langsmith_eval, "get_chat_model", _boom)

    with pytest.raises(RuntimeError):
        langsmith_eval.run_langsmith_eval("openai")

    assert settings.llm_provider == "anthropic"
