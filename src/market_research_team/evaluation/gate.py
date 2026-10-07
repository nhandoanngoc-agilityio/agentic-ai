"""Release gate: absolute thresholds, baseline tolerances and case regressions.

Pure: no LLM, no network. Config lives in the committed `evals/gate.toml`;
the baseline (the last approved version's metrics, per provider) lives in the
committed `evals/baseline.json`. A reviewed commit of that file is the
promotion step.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from market_research_team.evaluation.metrics import EvalMetrics, Price

MAX_UNJUDGED_FRACTION = 0.5
_EPSILON = 1e-9


class GateConfigError(ValueError):
    """gate.toml or baseline.json is missing, malformed, or refuses an update."""


@dataclass(frozen=True)
class Thresholds:
    task_success_min: float
    quality_min: float
    tool_accuracy_min: float
    safety_min: float
    latency_p95_ms_max: float
    cost_per_run_usd_max: float


@dataclass(frozen=True)
class Tolerance:
    task_success_max_drop: float
    quality_max_drop: float
    tool_accuracy_max_drop: float
    latency_max_increase: float
    cost_max_increase: float
    model_calls_max_increase: float = 0.25


@dataclass(frozen=True)
class GateConfig:
    baseline_min_repeats: int
    judge_provider: str
    judge_model: str
    thresholds: Thresholds
    tolerance: Tolerance
    prices: dict[str, Price]


@dataclass
class BaselineEntry:
    agent_version: str
    git_sha: str
    created_at: str
    repeats: int
    # What produced the quality scores (judges.judge_id): source, judge model, prompts.
    judge_id: str
    manifest: dict[str, Any]
    metrics: EvalMetrics


@dataclass
class GateCheck:
    name: str
    kind: str  # "absolute" | "baseline" | "case_regression"
    value: float | None
    limit: str
    passed: bool


@dataclass
class GateReport:
    provider: str
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_gate_config(path: Path) -> GateConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return GateConfig(
            baseline_min_repeats=int(raw["baseline_min_repeats"]),
            judge_provider=str(raw["judge"]["provider"]),
            judge_model=str(raw["judge"]["model"]),
            thresholds=Thresholds(**raw["thresholds"]),
            tolerance=Tolerance(**raw["tolerance"]),
            prices={
                model: Price(float(value["input"]), float(value["output"]))
                for model, value in raw.get("prices", {}).items()
            },
        )
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as exc:
        raise GateConfigError(f"Invalid gate config {path}: {exc}") from exc


def _at_least(name: str, value: float | None, floor: float) -> GateCheck:
    passed = value is not None and value >= floor - _EPSILON
    return GateCheck(name, "absolute", value, f">= {floor:g}", passed)


def _at_most(name: str, value: float | None, ceiling: float) -> GateCheck:
    passed = value is not None and value <= ceiling + _EPSILON
    limit = f"<= {ceiling:g}" + ("" if value is not None else " (no samples)")
    return GateCheck(name, "absolute", value, limit, passed)


def _quality_check(m: EvalMetrics, floor: float) -> GateCheck:
    if m.quality is None or m.unjudged_fraction > MAX_UNJUDGED_FRACTION:
        limit = (
            f">= {floor:g}; unjudged {m.unjudged_fraction:.0%} (max {MAX_UNJUDGED_FRACTION:.0%})"
        )
        return GateCheck("quality", "absolute", m.quality, limit, False)
    return _at_least("quality", m.quality, floor)


def _absolute_checks(m: EvalMetrics, t: Thresholds) -> list[GateCheck]:
    return [
        _at_least("task_success", m.task_success, t.task_success_min),
        _quality_check(m, t.quality_min),
        _at_least("tool_accuracy", m.tool_accuracy, t.tool_accuracy_min),
        _at_least("safety", m.safety, t.safety_min),
        _at_most("latency_p95_ms", m.latency_p95_ms, t.latency_p95_ms_max),
        _at_most("cost_per_run_usd", m.cost_per_run_usd, t.cost_per_run_usd_max),
    ]


def _max_drop(name: str, value: float | None, base: float | None, drop: float) -> GateCheck | None:
    if value is None or base is None:
        return None
    passed = value >= base - drop - _EPSILON
    return GateCheck(name, "baseline", value, f"drop <= {drop:g} (baseline {base:.4g})", passed)


def _max_increase(
    name: str, value: float | None, base: float | None, increase: float
) -> GateCheck | None:
    if value is None or base is None or base <= 0:
        return None
    passed = value <= base * (1 + increase) + _EPSILON
    return GateCheck(name, "baseline", value, f"<= +{increase:.0%} (baseline {base:.4g})", passed)


def _latency_warning(m: EvalMetrics, b: EvalMetrics, increase: float) -> str | None:
    """Wall-clock drift vs the baseline, reported but not blocking.

    Run time moves with provider response time as much as with the agent: a
    slow hour shifted every run (median +42%) while the agent did less work.
    Work is gated by `model_calls_per_run` and cost; time by the absolute
    `latency_p95_ms_max` cap. This says when the two disagree.
    """

    base = b.latency_median_ms if b.latency_median_ms is not None else b.latency_p95_ms
    if m.latency_median_ms is None or not base or m.latency_median_ms <= base * (1 + increase):
        return None
    which = "median" if b.latency_median_ms is not None else "p95 (no median recorded)"
    drift = m.latency_median_ms / base - 1
    calls = ""
    if usage_comparable(m, b) and m.model_calls_per_run is not None and b.model_calls_per_run:
        call_drift = m.model_calls_per_run / b.model_calls_per_run - 1
        calls = f"; model calls per run {call_drift:+.0%}"
    return (
        f"median run latency {drift:+.0%} vs baseline {which}{calls}. Not blocking: "
        "unchanged work points to provider response time."
    )


def usage_comparable(m: EvalMetrics, b: EvalMetrics) -> bool:
    """Cost and model calls compare only when both were recorded the same way."""

    return m.usage_accounting == b.usage_accounting


def _baseline_checks(m: EvalMetrics, b: EvalMetrics, tol: Tolerance) -> list[GateCheck]:
    candidates = [
        _max_drop("task_success", m.task_success, b.task_success, tol.task_success_max_drop),
        _max_drop("quality", m.quality, b.quality, tol.quality_max_drop),
        _max_drop("tool_accuracy", m.tool_accuracy, b.tool_accuracy, tol.tool_accuracy_max_drop),
    ]
    if usage_comparable(m, b):
        candidates += [
            _max_increase(
                "model_calls_per_run",
                m.model_calls_per_run,
                b.model_calls_per_run,
                tol.model_calls_max_increase,
            ),
            _max_increase(
                "cost_per_run_usd", m.cost_per_run_usd, b.cost_per_run_usd, tol.cost_max_increase
            ),
        ]
    return [check for check in candidates if check is not None]


def _case_regressions(m: EvalMetrics, b: EvalMetrics) -> list[GateCheck]:
    return [
        GateCheck(
            key, "case_regression", m.case_pass_rate[key], "baseline passed every repeat", False
        )
        for key, base_rate in sorted(b.case_pass_rate.items())
        if base_rate == 1.0 and m.case_pass_rate.get(key) == 0.0
    ]


def evaluate_gate(
    provider: str,
    candidate: EvalMetrics,
    baseline: BaselineEntry | None,
    config: GateConfig,
    *,
    judge_id: str,
) -> GateReport:
    checks = _absolute_checks(candidate, config.thresholds)
    warnings: list[str] = []
    if candidate.unjudged_fraction > 0:
        warnings.append(
            f"{candidate.unjudged_fraction:.0%} of judgeable cases are unjudged; "
            "see judge_error in the case details."
        )
    if baseline is None:
        warnings.append(f"No baseline for provider {provider!r}: absolute checks only.")
    else:
        checks += _baseline_checks(candidate, baseline.metrics, config.tolerance)
        latency = _latency_warning(
            candidate, baseline.metrics, config.tolerance.latency_max_increase
        )
        if latency:
            warnings.append(latency)
        if not usage_comparable(candidate, baseline.metrics):
            warnings.append(
                f"baseline recorded model usage differently (accounting "
                f"v{baseline.metrics.usage_accounting}, this run "
                f"v{candidate.usage_accounting}): cost and model calls are not compared "
                "until the baseline is refreshed; the absolute cost cap still applies."
            )
        elif baseline.metrics.model_calls_per_run is None:
            warnings.append(
                "baseline has no model-call count (recorded before it existed): agent work "
                "is not compared until the baseline is refreshed."
            )
        checks += _case_regressions(candidate, baseline.metrics)
        if baseline.judge_id != judge_id:
            warnings.append(
                f"judge changed since the baseline ({baseline.judge_id} -> {judge_id}): "
                "quality comparison is not like-for-like."
            )
    return GateReport(provider, all(c.passed for c in checks), checks, warnings)


def load_baseline(path: Path) -> dict[str, BaselineEntry]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            provider: BaselineEntry(**{**entry, "metrics": EvalMetrics(**entry["metrics"])})
            for provider, entry in raw.items()
        }
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
        raise GateConfigError(f"Invalid baseline {path}: {exc}") from exc


def save_baseline(path: Path, provider: str, entry: BaselineEntry) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing[provider] = asdict(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def baseline_refusal(report: GateReport, metrics: EvalMetrics, config: GateConfig) -> str | None:
    """Why the baseline must not be updated from this run, or None if it may."""

    if not report.passed:
        return f"gate failed for {report.provider!r}; baseline not updated."
    if metrics.repeats < config.baseline_min_repeats:
        return (
            f"baseline needs --repeats >= {config.baseline_min_repeats} "
            f"(this run had {metrics.repeats}); baseline not updated."
        )
    return None
