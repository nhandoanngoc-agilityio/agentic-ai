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
    candidate = replace(GOOD, quality=0.81, latency_p95_ms=74000.0, cost_per_run_usd=0.119)

    report = evaluate_gate("anthropic", candidate, _baseline(), config, judge_id="h")

    assert report.passed, _failed(report)


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ({"task_success": 0.94}, "task_success"),
        ({"quality": 0.79}, "quality"),
        ({"tool_accuracy": 0.949}, "tool_accuracy"),
        ({"latency_p95_ms": 76000.0}, "latency_p95_ms"),
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
