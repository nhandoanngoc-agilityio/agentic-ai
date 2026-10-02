"""Metric computation from eval results + a scoped audit log (pure, no LLM)."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from market_research_team.evaluation.metrics import (
    EvalMetrics,
    MissingPriceError,
    Price,
    compute_metrics,
    p95,
)
from market_research_team.evaluation.results import EvalResult

PRICES = {"m": Price(input_per_mtok=2.0, output_per_mtok=10.0)}


def _audit(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n", encoding="utf-8")
    return path


def _r(category, name, passed, judge=None, repeat=0):
    return EvalResult(category, name, passed, "d", "p", judge, repeat)


def test_p95_nearest_rank():
    assert p95([]) is None
    assert p95([5.0]) == 5.0
    assert p95([float(v) for v in range(1, 21)]) == 19.0


def test_rates_quality_and_case_pass_rate(tmp_path: Path):
    results = [
        _r("query_rewrite", "a", True, 0.8, 0),
        _r("query_rewrite", "a", False, 0.6, 1),
        _r("reporting", "b", True, None, 0),
        _r("reporting", "b", True, 1.0, 1),
        _r("retrieval", "c", True, None, 0),
        _r("retrieval", "c", True, None, 1),
        _r("safety", "s", True, None, 0),
        _r("safety", "s", False, None, 1),
    ]

    m = compute_metrics(results, tmp_path / "missing.jsonl", model="m", prices=PRICES)

    assert m.task_success == pytest.approx(5 / 6)
    assert m.safety == 0.5
    assert m.quality == pytest.approx((0.8 + 0.6 + 1.0) / 3)
    assert m.quality_by_category == {"query_rewrite": pytest.approx(0.7), "reporting": 1.0}
    assert m.unjudged_fraction == pytest.approx(1 / 4)
    assert m.case_pass_rate["query_rewrite/a"] == 0.5
    assert m.case_pass_rate["safety/s"] == 0.5
    assert m.repeats == 2


def test_tool_accuracy_combines_call_outcomes_and_selection(tmp_path: Path):
    audit = _audit(
        tmp_path / "a.jsonl",
        [
            {"event": "tool_call", "outcome": "ok"},
            {"event": "tool_call", "outcome": "ok"},
            {"event": "tool_call", "outcome": "error"},
            {"event": "tool_call", "outcome": "unknown_tool"},
        ],
    )
    results = [_r("tool_selection", "x", True), _r("tool_selection", "x", False, repeat=1)]

    m = compute_metrics(results, audit, model="m", prices=PRICES)

    assert m.tool_accuracy == pytest.approx(0.5 * 0.5)


def test_latency_and_cost_per_graph_run(tmp_path: Path):
    audit = _audit(
        tmp_path / "a.jsonl",
        [
            {"event": "run_latency", "thread_id": "eval-a-r0", "duration_ms": 1000},
            {"event": "run_latency", "thread_id": "eval-a-r0", "duration_ms": 500},
            {"event": "run_latency", "thread_id": "eval-b-r0", "duration_ms": 3000},
            {"event": "run_latency", "thread_id": None, "duration_ms": 99999},
            {
                "event": "llm_usage",
                "thread_id": "eval-a-r0",
                "input_tokens": 1000,
                "output_tokens": 500,
            },
            {
                "event": "llm_usage",
                "thread_id": None,
                "input_tokens": 10**9,
                "output_tokens": 10**9,
            },
        ],
    )

    m = compute_metrics([], audit, model="m", prices=PRICES)

    assert m.latency_p95_ms == 3000.0
    # run a: 1000*2/1e6 + 500*10/1e6 = 0.007; run b: 0.0 -> mean 0.0035
    assert m.cost_per_run_usd == pytest.approx(0.0035)


def test_no_graph_runs_means_no_latency_or_cost(tmp_path: Path):
    m = compute_metrics([], tmp_path / "none.jsonl", model="m", prices=PRICES)

    assert m.latency_p95_ms is None and m.cost_per_run_usd is None
    assert m.task_success == 0.0 and m.safety == 0.0 and m.tool_accuracy == 0.0


@pytest.mark.parametrize("prices", [{}, {"m": Price(0.0, 0.0)}])
def test_missing_or_zero_price_raises(tmp_path: Path, prices):
    with pytest.raises(MissingPriceError, match="'m'"):
        compute_metrics([], tmp_path / "none.jsonl", model="m", prices=prices)


def test_metrics_round_trip_through_dict():
    m = EvalMetrics(1.0, 0.9, {"reporting": 0.9}, 0.0, 1.0, 1.0, 10.0, 0.01, {"a/b": 1.0}, 3)

    assert EvalMetrics(**asdict(m)) == m
