"""CLI for the prompt evaluation regression suite and release gate.

Unlike `pytest`, this makes real LLM calls against a live provider and
checks structural/grounded properties of the output — run it on demand
(before a release, after changing a prompt or switching models), not as
part of the hermetic test suite. Requires the vector store to be seeded
(`python scripts/seed_vectorstore.py`) for the full-pipeline case.

    python scripts/run_evals.py                  # current LLM_PROVIDER
    python scripts/run_evals.py --provider openai
    python scripts/run_evals.py --compare         # anthropic AND openai, side by side
    python scripts/run_evals.py --repeats 3                    # average 3 runs (release check)
    python scripts/run_evals.py --repeats 3 --update-baseline  # promote: rewrite the baseline

Every run ends with a release-gate verdict per provider (thresholds and
tolerances in evals/gate.toml, last approved metrics in evals/baseline.json).

Exit codes: 0 gate passed, 1 gate failed (or cache regression), 2 config/prerequisite error.
"""

import argparse
import sys
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from market_research_team.config import settings
from market_research_team.evaluation.cache_regression import CacheRegressionReport, run_cold_vs_warm
from market_research_team.evaluation.gate import (
    BaselineEntry,
    GateConfigError,
    GateReport,
    baseline_refusal,
    evaluate_gate,
    load_baseline,
    load_gate_config,
    save_baseline,
)
from market_research_team.evaluation.golden_dataset import REGRESSION_CASES
from market_research_team.evaluation.judges import judge_id
from market_research_team.evaluation.langsmith_eval import (
    LangSmithEvalSummary,
    langsmith_api_key,
    run_langsmith_eval,
)
from market_research_team.evaluation.metrics import (
    EvalMetrics,
    MissingPriceError,
    compute_metrics,
    require_price,
)
from market_research_team.evaluation.offline_eval import (
    EvalResult,
    run_all,
    save_results,
    scoped_eval_run,
)
from market_research_team.llm import get_chat_model, model_id_for
from market_research_team.security.audit import (
    fallback_counts,
    latency_summary,
    satisfaction_rate,
    token_usage_summary,
    tool_call_summary,
)
from market_research_team.versioning import build_manifest, diff_manifests, git_sha

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "data" / "eval_results"
_DEFAULT_GATE_CONFIG = _ROOT / "evals" / "gate.toml"
_DEFAULT_BASELINE = _ROOT / "evals" / "baseline.json"

EXIT_PASS, EXIT_GATE_FAILED, EXIT_CONFIG_ERROR = 0, 1, 2


def _print_report(results: list[EvalResult]) -> None:
    by_provider: dict[str, list[EvalResult]] = {}
    for result in results:
        by_provider.setdefault(result.provider, []).append(result)

    for provider, provider_results in by_provider.items():
        passed = sum(1 for result in provider_results if result.passed)
        total = len(provider_results)
        print(f"\n=== {provider} — {passed}/{total} passed ===")
        for result in provider_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.category}/{result.case_name}: {result.detail}")


def _langsmith_prereq_error(langsmith_flag: bool, api_key: str | None) -> str | None:
    if langsmith_flag and not api_key:
        return "--langsmith requires LANGSMITH_API_KEY (in your shell or in .env)"
    return None


def _print_langsmith_report(summaries: list[LangSmithEvalSummary]) -> None:
    print("\n=== LangSmith experiments ===")
    for summary in summaries:
        print(
            f"  {summary.category}: pass_rate={summary.pass_rate:.2f} "
            f"experiment={summary.experiment_name!r}"
        )


def _print_cache_regression_report(report: CacheRegressionReport) -> None:
    print(f"\n=== Cache regression — hit rate {report.cache_hit_rate:.2f} ===")
    if not report.regressions:
        print("  No regressions: caching did not change any case's outcome.")
        return
    for _cold, warm in report.regressions:
        print(
            f"  [REGRESSION] {warm.category}/{warm.case_name}: cold=PASS warm=FAIL ({warm.detail})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run and gate anthropic AND openai (needs both API keys set).",
    )
    parser.add_argument(
        "--cache-regression",
        action="store_true",
        help=(
            "Run the suite cold then warm and fail if caching changes any "
            "case's outcome (single provider, extra LLM calls)."
        ),
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help=(
            "Also run LangSmith experiments; their judge scores replace the local "
            "quality metric (requires LANGSMITH_API_KEY)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the suite N times and average (use 3 for a release).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="After a passing gate with enough repeats, rewrite the baseline.",
    )
    parser.add_argument("--gate-config", type=Path, default=_DEFAULT_GATE_CONFIG)
    parser.add_argument("--baseline", type=Path, default=_DEFAULT_BASELINE)
    return parser


def _providers(args: argparse.Namespace) -> list[str]:
    if args.compare and not args.cache_regression:
        return ["anthropic", "openai"]
    return [args.provider or settings.llm_provider]


def manifest_for(provider: str) -> dict[str, Any]:
    """The manifest as `provider` sees it -- in --compare the process-wide
    provider setting is not the one being gated."""

    original = settings.llm_provider
    settings.llm_provider = provider  # type: ignore[assignment]
    try:
        return build_manifest()
    finally:
        settings.llm_provider = original


def run_provider(
    provider: str, repeats: int, run_dir: Path, judge_llm: BaseChatModel | None
) -> tuple[list[EvalResult], Path]:
    provider_dir = run_dir / provider
    results: list[EvalResult] = []
    with scoped_eval_run(provider_dir):
        for repeat in range(repeats):
            results += run_all(provider, repeat=repeat, judge_llm=judge_llm)  # type: ignore[arg-type]
    return results, provider_dir / "audit.jsonl"


def _run_cache_regression(
    provider: str, run_dir: Path, judge_llm: BaseChatModel | None
) -> tuple[list[EvalResult], Path, bool]:
    """Cold then warm, each in its own scoped directory; only the warm pass is
    judged and gated."""

    provider_dir = run_dir / provider
    report = run_cold_vs_warm(provider, judge_llm=judge_llm, run_dir=provider_dir)
    _print_cache_regression_report(report)
    warm_audit = report.warm_audit_path or provider_dir / "warm" / "audit.jsonl"
    return report.warm_results, warm_audit, bool(report.regressions)


def apply_langsmith_quality(
    metrics: EvalMetrics, summaries: list[LangSmithEvalSummary]
) -> EvalMetrics:
    judged = [s for s in summaries if s.mean_judge_score is not None and s.judged_count]
    total = sum(s.judged_count for s in judged)
    rows = sum(s.judged_count if s.row_count is None else s.row_count for s in summaries)
    if not total:
        return replace(metrics, quality=None, quality_by_category={}, unjudged_fraction=1.0)
    return replace(
        metrics,
        quality=sum(s.mean_judge_score * s.judged_count for s in judged) / total,  # type: ignore[operator]
        quality_by_category={s.category: s.mean_judge_score for s in judged},  # type: ignore[misc]
        # Rows whose judge errored carry no score; count them so a mostly broken
        # judge cannot pass the quality check by omission.
        unjudged_fraction=1 - total / rows if rows else 1.0,
    )


def _print_audit_summaries(audit_path: Path) -> None:
    latencies = latency_summary(audit_path)
    if latencies:
        print("\n=== Latency (this run) ===")
        for name, stats in latencies.items():
            print(f"  {name}: mean={stats['mean_ms']:.0f}ms (n={stats['count']:.0f})")
    fallbacks = fallback_counts(audit_path)
    if fallbacks:
        print("\n=== Fallback events (this run) ===")
        for component, count in fallbacks.items():
            print(f"  {component}: {count}")
    token_usage = token_usage_summary(audit_path)
    if token_usage:
        print("\n=== Token usage (this run) ===")
        for component, tokens in token_usage.items():
            print(
                f"  {component}: input={tokens['input_tokens']} "
                f"output={tokens['output_tokens']} total={tokens['total_tokens']}"
            )
    tool_calls = tool_call_summary(audit_path)
    if tool_calls:
        print("\n=== Tool calls (this run) ===")
        for tool, stats in tool_calls.items():
            print(
                f"  {tool}: mean={stats['mean_ms']:.0f}ms (n={stats['count']}) "
                f"ok={stats['ok']} error={stats['error']} unknown_tool={stats['unknown_tool']}"
            )


def _print_gate_report(
    report: GateReport, candidate_manifest: dict[str, Any], baseline: BaselineEntry | None
) -> None:
    version = f"{candidate_manifest['release']}+{candidate_manifest['fingerprint']}"
    against = f"vs baseline {baseline.agent_version}" if baseline else "no baseline"
    print(f"\n=== Release gate: {report.provider}  (candidate {version} {against}) ===")
    if baseline:
        changes = diff_manifests(baseline.manifest, candidate_manifest)
        print("Changed since baseline:" + ("" if changes else " nothing"))
        for line in changes:
            print(f"  {line}")
    for warning in report.warnings:
        print(f"  WARNING {warning}")
    for check in report.checks:
        value = "n/a" if check.value is None else f"{check.value:.4g}"
        status = "PASS" if check.passed else "FAIL"
        print(f"  {status}  {check.kind:<15} {check.name:<40} {value:>10}  {check.limit}")
    failed = sum(not check.passed for check in report.checks)
    print("Gate PASSED" if report.passed else f"Gate FAILED ({failed} checks)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prereq_error = _langsmith_prereq_error(args.langsmith, langsmith_api_key())
    if prereq_error is not None:
        print(prereq_error, file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    providers = _providers(args)
    try:
        config = load_gate_config(args.gate_config)
        baselines = load_baseline(args.baseline)
        for provider in providers:  # before any paid call
            require_price(config.prices, model_id_for(provider))
    except (GateConfigError, MissingPriceError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG_ERROR

    run_dir = _RESULTS_DIR / datetime.now(UTC).strftime("eval_%Y%m%dT%H%M%SZ")
    print(f"Eval run {run_dir.name} (git {git_sha()}), repeats={args.repeats}")
    graph_cases = sum(1 for c in REGRESSION_CASES if not c.expectations.must_block)
    print(f"Regression cases: {len(REGRESSION_CASES)} ({graph_cases} run the full graph)")
    # One pinned judge for both paths: local judging, or LangSmith's llm_judge.
    judge_llm = get_chat_model(config.judge_provider, config.judge_model)
    local_judge = None if args.langsmith else judge_llm
    quality_judge_id = judge_id(
        "langsmith" if args.langsmith else "local", config.judge_provider, config.judge_model
    )

    all_results: list[EvalResult] = []
    gate_entries: list[dict[str, Any]] = []
    reports: list[GateReport] = []
    cache_regressed = False
    refused = False
    crashed = False
    try:
        for provider in providers:
            manifest = manifest_for(provider)
            version = f"{manifest['release']}+{manifest['fingerprint']}"
            print(f"\nEvaluating {provider}: agent {version}")
            if args.cache_regression:
                results, audit_path, regressed = _run_cache_regression(
                    provider, run_dir, local_judge
                )
                cache_regressed = cache_regressed or regressed
            else:
                results, audit_path = run_provider(provider, args.repeats, run_dir, local_judge)
            all_results += results
            _print_report(results)
            _print_audit_summaries(audit_path)

            metrics = compute_metrics(
                results, audit_path, model=model_id_for(provider), prices=config.prices
            )
            if args.langsmith:
                with scoped_eval_run(run_dir / provider / "langsmith"):
                    summaries = run_langsmith_eval(provider, judge_llm=judge_llm)  # type: ignore[arg-type]
                _print_langsmith_report(summaries)
                metrics = apply_langsmith_quality(metrics, summaries)

            baseline = baselines.get(provider)
            report = evaluate_gate(provider, metrics, baseline, config, judge_id=quality_judge_id)
            reports.append(report)
            gate_entries.append({**asdict(report), "agent_version": version, "manifest": manifest})
            _print_gate_report(report, manifest, baseline)

            if args.update_baseline:
                refusal = baseline_refusal(report, metrics, config)
                if refusal:
                    print(refusal, file=sys.stderr)
                    refused = True
                else:
                    save_baseline(
                        args.baseline,
                        provider,
                        BaselineEntry(
                            agent_version=version,
                            git_sha=git_sha(),
                            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                            repeats=metrics.repeats,
                            judge_id=quality_judge_id,
                            manifest=manifest,
                            metrics=metrics,
                        ),
                    )
                    print(
                        f"Baseline updated for {provider} in {args.baseline}; commit it to promote."
                    )
    except Exception:
        # Infrastructure trouble (unseeded vector store, provider outage, missing
        # key) is not a candidate regression: report it as exit 2, keep partial results.
        traceback.print_exc()
        print("Eval run aborted by an unexpected error (see traceback).", file=sys.stderr)
        crashed = True
    finally:
        output_path = save_results(
            all_results, run_dir, gate=gate_entries, manifest=manifest_for(providers[0])
        )
        print(f"\nSaved results to {output_path}")

    rate, rating_count = satisfaction_rate()
    print(f"User satisfaction (production log): {rate:.0%} ({rating_count} ratings)")

    if crashed or refused:
        return EXIT_CONFIG_ERROR
    if cache_regressed or any(not r.passed for r in reports):
        return EXIT_GATE_FAILED
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
