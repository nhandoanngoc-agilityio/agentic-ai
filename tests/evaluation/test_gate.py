"""Release-gate checks against thresholds and a baseline (pure)."""

from dataclasses import replace
from pathlib import Path

import pytest

from market_research_team.evaluation.gate import (
    BaselineEntry,
    GateConfigError,
    evaluate_gate,
    load_gate_config,
)
from market_research_team.evaluation.metrics import EvalMetrics

GATE_TOML = """
baseline_min_repeats = 3

[judge]
provider = "anthropic"
model = "claude-opus-5"

[thresholds]
task_success_min     = 0.90
quality_min          = 0.70
tool_accuracy_min    = 0.95
safety_min           = 1.00
latency_p95_ms_max   = 120000
cost_per_run_usd_max = 0.50

[tolerance]
task_success_max_drop  = 0.05
quality_max_drop       = 0.05
tool_accuracy_max_drop = 0.05
latency_max_increase   = 0.25
cost_max_increase      = 0.20

[prices."m"]
input = 2.0
output = 10.0
"""

GOOD = EvalMetrics(
    task_success=1.0,
    quality=0.85,
    quality_by_category={"reporting": 0.85},
    unjudged_fraction=0.0,
    tool_accuracy=1.0,
    safety=1.0,
    latency_p95_ms=60000.0,
    cost_per_run_usd=0.10,
    case_pass_rate={"reporting/r": 1.0, "analytics/a": 1.0},
    repeats=3,
    latency_median_ms=50000.0,
    model_calls_per_run=10.0,
)


@pytest.fixture
def config(tmp_path: Path):
    path = tmp_path / "gate.toml"
    path.write_text(GATE_TOML, encoding="utf-8")
    return load_gate_config(path)


def _baseline(metrics: EvalMetrics = GOOD, judge_hash: str = "h") -> BaselineEntry:
    return BaselineEntry("0.1.0+aaa", "sha", "2026-09-29T00:00:00Z", 3, judge_hash, {}, metrics)


def _failed(report) -> set[tuple[str, str]]:
    return {(c.kind, c.name) for c in report.checks if not c.passed}


def test_load_gate_config_reads_every_section(config):
    assert config.baseline_min_repeats == 3
    assert config.judge_model == "claude-opus-5"
    assert config.thresholds.cost_per_run_usd_max == 0.50
    assert config.tolerance.latency_max_increase == 0.25
    assert config.prices["m"].output_per_mtok == 10.0


@pytest.mark.parametrize(
    "text", ["not = [valid", "baseline_min_repeats = 3\n", GATE_TOML.replace("quality_min", "q")]
)
def test_malformed_config_raises_gate_config_error(tmp_path: Path, text: str):
    path = tmp_path / "gate.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(GateConfigError):
        load_gate_config(path)


def test_missing_config_file_raises_gate_config_error(tmp_path: Path):
    with pytest.raises(GateConfigError):
        load_gate_config(tmp_path / "absent.toml")


def test_good_candidate_without_baseline_passes_with_warning(config):
    report = evaluate_gate("anthropic", GOOD, None, config, judge_id="h")

    assert report.passed
    assert {c.kind for c in report.checks} == {"absolute"}
    assert any("No baseline" in w for w in report.warnings)


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ({"task_success": 0.8}, "task_success"),
        ({"quality": 0.5}, "quality"),
        ({"tool_accuracy": 0.9}, "tool_accuracy"),
        ({"safety": 0.99}, "safety"),
        ({"latency_p95_ms": 130000.0}, "latency_p95_ms"),
        ({"cost_per_run_usd": 0.6}, "cost_per_run_usd"),
        ({"latency_p95_ms": None}, "latency_p95_ms"),
        ({"cost_per_run_usd": None}, "cost_per_run_usd"),
        ({"unjudged_fraction": 0.6}, "quality"),
        ({"quality": None, "unjudged_fraction": 1.0}, "quality"),
    ],
)
def test_each_absolute_failure(config, change, name):
    report = evaluate_gate("anthropic", replace(GOOD, **change), None, config, judge_id="h")

    assert not report.passed
    assert ("absolute", name) in _failed(report)


def test_baseline_drop_within_tolerance_passes(config):
    candidate = replace(GOOD, quality=0.81, latency_median_ms=62000.0, cost_per_run_usd=0.119)

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert report.passed, _failed(report)


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ({"task_success": 0.94}, "task_success"),
        ({"quality": 0.79}, "quality"),
        ({"tool_accuracy": 0.949}, "tool_accuracy"),
        ({"model_calls_per_run": 12.6}, "model_calls_per_run"),
        ({"cost_per_run_usd": 0.121}, "cost_per_run_usd"),
    ],
)
def test_each_baseline_regression(config, change, name):
    baseline = _baseline(replace(GOOD, task_success=1.0, tool_accuracy=1.0))
    candidate = replace(GOOD, **change)

    report = evaluate_gate("anthropic", candidate, baseline, config, judge_id="h")

    assert ("baseline", name) in _failed(report)


def test_case_regression_is_named_and_new_cases_are_not_regressions(config):
    candidate = replace(
        GOOD, case_pass_rate={"reporting/r": 0.0, "analytics/a": 1.0, "safety/new": 0.0}
    )

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert ("case_regression", "reporting/r") in _failed(report)
    assert not any(c.name == "safety/new" for c in report.checks)


def test_partial_case_failure_is_not_a_case_regression(config):
    candidate = replace(GOOD, case_pass_rate={"reporting/r": 0.34, "analytics/a": 1.0})

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert not any(c.kind == "case_regression" for c in report.checks)


def test_judge_prompt_change_warns(config):
    report = evaluate_gate("anthropic", GOOD, _baseline(judge_hash="old"), config, judge_id="new")

    assert any("judge changed" in w for w in report.warnings)


def test_load_baseline_absent_file_is_empty(tmp_path: Path):
    from market_research_team.evaluation.gate import load_baseline

    assert load_baseline(tmp_path / "baseline.json") == {}


def test_save_then_load_round_trips_and_keeps_other_providers(tmp_path: Path):
    from market_research_team.evaluation.gate import load_baseline, save_baseline

    path = tmp_path / "evals" / "baseline.json"
    save_baseline(path, "openai", _baseline(replace(GOOD, quality=0.7)))
    save_baseline(path, "anthropic", _baseline())

    loaded = load_baseline(path)

    assert set(loaded) == {"anthropic", "openai"}
    assert loaded["anthropic"].metrics == GOOD
    assert loaded["openai"].metrics.quality == 0.7
    assert path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize("text", ["{not json", '{"anthropic": {"agent_version": "x"}}'])
def test_malformed_baseline_raises_gate_config_error(tmp_path: Path, text: str):
    from market_research_team.evaluation.gate import load_baseline

    path = tmp_path / "baseline.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(GateConfigError):
        load_baseline(path)


def test_baseline_refusal_rules(config):
    from market_research_team.evaluation.gate import GateReport, baseline_refusal

    passed = GateReport("anthropic", True)
    failed = GateReport("anthropic", False)

    assert baseline_refusal(passed, GOOD, config) is None
    assert "gate failed" in baseline_refusal(failed, GOOD, config)
    assert "repeats" in baseline_refusal(passed, replace(GOOD, repeats=1), config)


def test_unjudged_cases_produce_a_warning(config):
    report = evaluate_gate(
        "anthropic", replace(GOOD, unjudged_fraction=0.25), None, config, judge_id="h"
    )

    assert any("unjudged" in w and "judge_error" in w for w in report.warnings)


def test_committed_gate_prices_every_default_agent_model() -> None:
    """A provider whose default model has no price makes `run_evals.py` exit 2
    before spending anything -- which is what the default Anthropic path did
    when its price entry was commented out."""

    from market_research_team.config import Settings
    from market_research_team.evaluation.metrics import require_price

    config = load_gate_config(Path(__file__).parents[2] / "evals" / "gate.toml")
    defaults = Settings.model_fields
    for model in (defaults["anthropic_model"].default, defaults["openai_model"].default):
        assert require_price(config.prices, model).input_per_mtok > 0


def test_one_slow_run_does_not_fail_the_baseline_latency_check(config):
    """p95 of ~9 runs is the slowest run: a slow provider response can move it
    +46% with no change in the agent. The median doesn't move, so it passes;
    the absolute p95 cap still bounds the slow run."""

    candidate = replace(GOOD, latency_p95_ms=95000.0)  # +58% p95, same median

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert report.passed, _failed(report)
    assert not any(c.name == "latency_p95_ms" and c.kind == "baseline" for c in report.checks)
    capped = evaluate_gate(
        "anthropic", replace(GOOD, latency_p95_ms=130000.0), _baseline(), config, judge_id="h"
    )
    assert ("absolute", "latency_p95_ms") in _failed(capped)


def test_a_slow_provider_hour_warns_but_does_not_fail(config):
    """The failing run of 2026-10-07: every call slower, the agent doing less
    work. Time drift is reported; work decides."""

    candidate = replace(GOOD, latency_median_ms=71000.0, model_calls_per_run=9.0)  # +42%, -10%

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert report.passed, _failed(report)
    (warning,) = [w for w in report.warnings if "median run latency" in w]
    assert warning.startswith(
        "median run latency +42% vs baseline median; model calls per run -10%"
    )


def test_a_loop_fails_on_model_calls_even_at_unchanged_latency(config):
    candidate = replace(GOOD, model_calls_per_run=14.0)  # +40% work, same time

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert ("baseline", "model_calls_per_run") in _failed(report)


def test_a_baseline_without_new_metrics_warns_instead_of_comparing(config):
    legacy = _baseline(replace(GOOD, latency_median_ms=None, model_calls_per_run=None))

    report = evaluate_gate(
        "anthropic", replace(GOOD, latency_median_ms=80000.0), legacy, config, judge_id="h"
    )

    assert report.passed, _failed(report)
    assert not any(c.name == "model_calls_per_run" for c in report.checks)
    assert any("no model-call count" in w for w in report.warnings)
    assert any("vs baseline p95 (no median recorded)" in w for w in report.warnings)


def test_a_baseline_saved_before_medians_existed_still_loads(tmp_path: Path):
    import json
    from dataclasses import asdict

    from market_research_team.evaluation.gate import load_baseline

    entry = asdict(_baseline())
    del entry["metrics"]["latency_median_ms"]
    del entry["metrics"]["model_calls_per_run"]
    del entry["metrics"]["model_calls_by_component"]
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"openai": entry}), encoding="utf-8")

    metrics = load_baseline(path)["openai"].metrics
    assert metrics.latency_median_ms is None and metrics.model_calls_per_run is None


def test_cost_and_calls_are_not_compared_across_usage_accounting(config):
    """A baseline that recorded only plain `.invoke` calls under-counts cost and
    calls; comparing a full count against it would always look like +100%."""

    old_accounting = _baseline(replace(GOOD, usage_accounting=1))
    candidate = replace(GOOD, usage_accounting=2, cost_per_run_usd=0.40, model_calls_per_run=30.0)

    report = evaluate_gate("anthropic", candidate, old_accounting, config, judge_id="h")

    assert report.passed, _failed(report)
    assert not any(
        c.name in ("cost_per_run_usd", "model_calls_per_run") and c.kind == "baseline"
        for c in report.checks
    )
    assert any("recorded model usage differently" in w for w in report.warnings)
    over_cap = evaluate_gate(
        "anthropic",
        replace(candidate, cost_per_run_usd=0.6),
        old_accounting,
        config,
        judge_id="h",
    )
    assert ("absolute", "cost_per_run_usd") in _failed(over_cap)
