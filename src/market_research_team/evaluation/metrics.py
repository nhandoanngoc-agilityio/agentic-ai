"""Gate metrics from one provider's eval results plus its scoped audit log.

Pure: no LLM, no network. Results from every repeat are passed in together,
so rates average across repeats naturally. Latency and cost cover only graph
runs, i.e. audit events whose thread id starts with `GRAPH_RUN_PREFIX`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_research_team.evaluation.results import EvalResult

GRAPH_RUN_PREFIX = "eval-"
JUDGED_CATEGORIES = frozenset(
    {"query_rewrite", "supervisor_decision", "analytics", "reporting", "full_pipeline"}
)
# Categories scored by their own metric rather than task success.
_OWN_METRIC_CATEGORIES = frozenset({"safety", "tool_selection"})


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


class MissingPriceError(ValueError):
    """The active model has no (or a zero) price in gate.toml -- cost is never skipped."""


@dataclass
class EvalMetrics:
    task_success: float
    quality: float | None
    quality_by_category: dict[str, float]
    unjudged_fraction: float
    tool_accuracy: float
    safety: float
    latency_p95_ms: float | None
    cost_per_run_usd: float | None
    case_pass_rate: dict[str, float]
    repeats: int


def require_price(prices: dict[str, Price], model: str) -> Price:
    price = prices.get(model)
    if price is None or (price.input_per_mtok == 0 and price.output_per_mtok == 0):
        raise MissingPriceError(
            f"No price for model {model!r} in gate.toml [prices]; add its USD per 1M tokens."
        )
    return price


def p95(values: list[float]) -> float | None:
    """Nearest-rank 95th percentile; with few samples this is the max (conservative)."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _rate(results: list[EvalResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _quality(results: list[EvalResult]) -> tuple[float | None, dict[str, float], float]:
    judgeable = [r for r in results if r.category in JUDGED_CATEGORIES]
    scored = [(r.category, r.judge_score) for r in judgeable if r.judge_score is not None]
    unjudged = 1 - len(scored) / len(judgeable) if judgeable else 1.0
    by_category: dict[str, list[float]] = {}
    for category, score in scored:
        by_category.setdefault(category, []).append(score)
    quality = sum(score for _, score in scored) / len(scored) if scored else None
    means = {cat: sum(v) / len(v) for cat, v in sorted(by_category.items())}
    return quality, means, unjudged


def _is_graph_run(event: dict[str, Any]) -> bool:
    return str(event.get("thread_id") or "").startswith(GRAPH_RUN_PREFIX)


def compute_metrics(
    results: list[EvalResult], audit_path: Path, *, model: str, prices: dict[str, Price]
) -> EvalMetrics:
    price = require_price(prices, model)
    events = read_audit(audit_path)

    task_results = [r for r in results if r.category not in _OWN_METRIC_CATEGORIES]
    quality, quality_by_category, unjudged = _quality(results)

    tool_events = [e for e in events if e.get("event") == "tool_call"]
    ok_rate = (
        sum(e.get("outcome") == "ok" for e in tool_events) / len(tool_events)
        if tool_events
        else 0.0
    )
    selection = [r for r in results if r.category == "tool_selection"]
    tool_accuracy = ok_rate * (_rate(selection) if selection else 1.0)

    # One graph run = one thread; its latency is the sum of its segments
    # (the initial run plus any approval resumes).
    run_ms: dict[str, float] = {}
    for e in events:
        if e.get("event") == "run_latency" and _is_graph_run(e):
            run_ms[e["thread_id"]] = run_ms.get(e["thread_id"], 0.0) + float(
                e.get("duration_ms", 0.0)
            )
    run_cost = {thread_id: 0.0 for thread_id in run_ms}
    for e in events:
        if e.get("event") == "llm_usage" and e.get("thread_id") in run_cost:
            run_cost[e["thread_id"]] += (
                int(e.get("input_tokens", 0)) * price.input_per_mtok
                + int(e.get("output_tokens", 0)) * price.output_per_mtok
            ) / 1_000_000

    case_results: dict[str, list[EvalResult]] = {}
    for r in results:
        case_results.setdefault(f"{r.category}/{r.case_name}", []).append(r)

    return EvalMetrics(
        task_success=_rate(task_results),
        quality=quality,
        quality_by_category=quality_by_category,
        unjudged_fraction=unjudged,
        tool_accuracy=tool_accuracy,
        safety=_rate([r for r in results if r.category == "safety"]),
        latency_p95_ms=p95(list(run_ms.values())),
        cost_per_run_usd=sum(run_cost.values()) / len(run_cost) if run_cost else None,
        case_pass_rate={key: _rate(rs) for key, rs in sorted(case_results.items())},
        repeats=len({r.repeat for r in results}),
    )
