"""Unit tests for the pure helper in scripts/run_evals.py.

Loaded by file path since `scripts/` isn't a package, same pattern as
tests/test_setup_env.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from market_research_team.config import settings
from market_research_team.evaluation.results import EvalResult

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_evals.py"
_spec = importlib.util.spec_from_file_location("run_evals", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evals)


def test_langsmith_prereq_error_when_flag_set_and_key_missing() -> None:
    assert run_evals._langsmith_prereq_error(True, None) == (
        "--langsmith requires LANGSMITH_API_KEY (in your shell or in .env)"
    )


def test_langsmith_prereq_error_none_when_flag_set_and_key_present() -> None:
    assert run_evals._langsmith_prereq_error(True, "sk-fake") is None


def test_langsmith_prereq_error_none_when_flag_not_set() -> None:
    assert run_evals._langsmith_prereq_error(False, None) is None


_GATE = """
baseline_min_repeats = 3
[judge]
provider = "anthropic"
model = "judge-model"
[thresholds]
task_success_min = 0.9
quality_min = 0.7
tool_accuracy_min = 0.95
safety_min = 1.0
latency_p95_ms_max = 120000
cost_per_run_usd_max = 0.5
[tolerance]
task_success_max_drop = 0.05
quality_max_drop = 0.05
tool_accuracy_max_drop = 0.05
latency_max_increase = 0.25
cost_max_increase = 0.2
[prices."test-model"]
input = 2.0
output = 10.0
[prices."openai-test-model"]
input = 1.0
output = 1.0
"""


def _fake_run_provider():
    def _run(provider, repeats, run_dir, judge_llm):
        provider_dir = run_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        audit = provider_dir / "audit.jsonl"
        lines = []
        results = []
        for r in range(repeats):
            tid = f"eval-full-r{r}"
            lines += [
                {"event": "tool_call", "outcome": "ok", "thread_id": None},
                {"event": "run_latency", "thread_id": tid, "duration_ms": 1000},
                {
                    "event": "llm_usage",
                    "thread_id": tid,
                    "input_tokens": 1000,
                    "output_tokens": 500,
                },
            ]
            results += [
                EvalResult("reporting", "r", True, "d", provider, 0.9, r),
                EvalResult("full_pipeline", "f", True, "d", provider, 0.9, r),
                EvalResult("tool_selection", "a", True, "d", provider, None, r),
                EvalResult("safety", "s", True, "d", provider, None, r),
            ]
        audit.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        return results, audit

    return _run


@pytest.fixture
def cli(tmp_path, monkeypatch):
    gate = tmp_path / "gate.toml"
    gate.write_text(_GATE, encoding="utf-8")
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_model", "test-model")
    monkeypatch.setattr(run_evals, "_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_evals, "get_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(run_evals, "run_provider", _fake_run_provider())
    base = ["--gate-config", str(gate), "--baseline", str(tmp_path / "baseline.json")]
    return base, tmp_path


def test_main_passes_and_writes_results_with_gate(cli):
    base, tmp_path = cli

    assert run_evals.main(base) == run_evals.EXIT_PASS

    results_file = next((tmp_path / "results").glob("eval_*/results.json"))
    data = json.loads(results_file.read_text(encoding="utf-8"))
    assert data["gate"][0]["passed"] is True


def test_main_exits_2_on_malformed_config(cli, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("nope = [", encoding="utf-8")

    assert run_evals.main(["--gate-config", str(bad)]) == run_evals.EXIT_CONFIG_ERROR


def test_main_exits_2_on_missing_price_before_running(cli, monkeypatch):
    base, _ = cli
    monkeypatch.setattr(settings, "anthropic_model", "unpriced-model")

    def must_not_run(*_a, **_k):
        raise AssertionError("paid run started before price check")

    monkeypatch.setattr(run_evals, "run_provider", must_not_run)

    assert run_evals.main(base) == run_evals.EXIT_CONFIG_ERROR


def test_main_exits_2_on_bad_repeats(cli):
    base, _ = cli

    assert run_evals.main([*base, "--repeats", "0"]) == run_evals.EXIT_CONFIG_ERROR


def test_update_baseline_refused_with_too_few_repeats(cli):
    base, tmp_path = cli

    code = run_evals.main([*base, "--update-baseline"])

    assert code == run_evals.EXIT_CONFIG_ERROR
    assert not (tmp_path / "baseline.json").exists()


def test_update_baseline_writes_entry_after_passing_3_repeats(cli):
    base, tmp_path = cli

    code = run_evals.main([*base, "--repeats", "3", "--update-baseline"])

    assert code == run_evals.EXIT_PASS
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["anthropic"]["repeats"] == 3
    assert saved["anthropic"]["agent_version"]


def test_gate_failure_exits_1(cli, monkeypatch):
    base, _ = cli

    def failing(provider, repeats, run_dir, judge_llm):
        results, audit = _fake_run_provider()(provider, repeats, run_dir, judge_llm)
        return [*results, EvalResult("safety", "x", False, "leak", provider)], audit

    monkeypatch.setattr(run_evals, "run_provider", failing)

    assert run_evals.main(base) == run_evals.EXIT_GATE_FAILED


def test_manifest_for_uses_the_given_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    manifest = run_evals.manifest_for("openai")

    assert manifest["components"]["model"]["provider"] == "openai"
    assert settings.llm_provider == "anthropic"


def test_apply_langsmith_quality_weights_by_judged_count():
    from market_research_team.evaluation.langsmith_eval import LangSmithEvalSummary
    from market_research_team.evaluation.metrics import EvalMetrics

    metrics = EvalMetrics(1.0, None, {}, 1.0, 1.0, 1.0, 1.0, 0.1, {}, 1)
    summaries = [
        LangSmithEvalSummary("reporting", "e1", 1.0, mean_judge_score=0.9, judged_count=1),
        LangSmithEvalSummary("analytics", "e2", 1.0, mean_judge_score=0.6, judged_count=2),
    ]

    updated = run_evals.apply_langsmith_quality(metrics, summaries)

    assert updated.quality == pytest.approx((0.9 + 1.2) / 3)
    assert updated.unjudged_fraction == 0.0  # row_count defaults to judged_count
    assert updated.quality_by_category == {"reporting": 0.9, "analytics": 0.6}


def test_apply_langsmith_quality_counts_rows_the_judge_missed():
    from market_research_team.evaluation.langsmith_eval import LangSmithEvalSummary
    from market_research_team.evaluation.metrics import EvalMetrics

    metrics = EvalMetrics(1.0, None, {}, 1.0, 1.0, 1.0, 1.0, 0.1, {}, 1)
    summaries = [
        LangSmithEvalSummary(
            "reporting", "e1", 1.0, mean_judge_score=0.9, judged_count=1, row_count=4
        )
    ]

    updated = run_evals.apply_langsmith_quality(metrics, summaries)

    assert updated.unjudged_fraction == pytest.approx(0.75)


def test_cache_regression_gates_the_warm_run_with_the_judge(cli, monkeypatch):
    from market_research_team.evaluation.cache_regression import CacheRegressionReport

    base, _ = cli
    seen: dict = {}

    def fake_cold_vs_warm(provider, *, judge_llm=None, run_dir=None):
        seen["judge_llm"] = judge_llm
        results, audit = _fake_run_provider()(provider, 1, run_dir.parent, judge_llm)
        return CacheRegressionReport(results, results, [], 0.5, warm_audit_path=audit)

    monkeypatch.setattr(run_evals, "run_cold_vs_warm", fake_cold_vs_warm)

    assert run_evals.main([*base, "--cache-regression"]) == run_evals.EXIT_PASS
    assert seen["judge_llm"] is not None


def test_langsmith_step_runs_inside_the_scoped_run_dir(cli, monkeypatch):
    base, tmp_path = cli
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    seen: dict = {}

    def fake_langsmith(provider, *, judge_llm=None):
        seen["audit"] = settings.audit_log_path
        seen["judge_llm"] = judge_llm
        return []

    monkeypatch.setattr(run_evals, "run_langsmith_eval", fake_langsmith)
    monkeypatch.setattr(run_evals, "get_chat_model", lambda *a, **k: "judge")

    run_evals.main([*base, "--langsmith"])

    assert str(seen["audit"]).startswith(str(tmp_path / "results"))
    assert seen["judge_llm"] == "judge"


def test_results_json_records_the_evaluated_providers_manifest(cli, monkeypatch):
    base, tmp_path = cli
    monkeypatch.setattr(settings, "openai_model", "openai-test-model")

    run_evals.main([*base, "--provider", "openai"])

    data = json.loads(next((tmp_path / "results").glob("eval_*/results.json")).read_text())
    assert data["manifest"]["components"]["model"]["provider"] == "openai"
    assert data["gate"][0]["manifest"]["components"]["model"]["model"] == "openai-test-model"


def test_unexpected_error_exits_2_and_keeps_partial_results(cli, monkeypatch):
    base, tmp_path = cli

    def boom(*_a, **_k):
        raise RuntimeError("vector store not seeded")

    monkeypatch.setattr(run_evals, "run_provider", boom)

    assert run_evals.main(base) == run_evals.EXIT_CONFIG_ERROR
    assert next((tmp_path / "results").glob("eval_*/results.json")).exists()


def test_langsmith_key_in_dotenv_passes_the_prereq_check(cli, monkeypatch):
    base, _ = cli
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(settings, "langsmith_api_key", "lsv2-from-dotenv")
    called: dict = {}

    def fake_langsmith(provider, *, judge_llm=None):
        called["yes"] = True
        return []

    monkeypatch.setattr(run_evals, "run_langsmith_eval", fake_langsmith)

    code = run_evals.main([*base, "--langsmith"])

    assert code != run_evals.EXIT_CONFIG_ERROR
    assert called["yes"]
