"""CLI for the prompt evaluation regression suite.

Unlike `pytest`, this makes real LLM calls against a live provider and
checks structural/grounded properties of the output — run it on demand
(before a release, after changing a prompt or switching models), not as
part of the hermetic test suite. Requires the vector store to be seeded
(`python scripts/seed_vectorstore.py`) for the full-pipeline case.

    python scripts/run_evals.py                  # current LLM_PROVIDER
    python scripts/run_evals.py --provider openai
    python scripts/run_evals.py --compare         # anthropic AND openai, side by side
"""

import argparse
import os
import sys
from pathlib import Path

from market_research_team.evaluation.langsmith_eval import LangSmithEvalSummary, run_langsmith_eval
from market_research_team.evaluation.offline_eval import EvalResult, run_all, save_results

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "eval_results"


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
        return "--langsmith requires LANGSMITH_API_KEY to be set"
    return None


def _print_langsmith_report(summaries: list[LangSmithEvalSummary]) -> None:
    print("\n=== LangSmith experiments ===")
    for summary in summaries:
        print(
            f"  {summary.category}: pass_rate={summary.pass_rate:.2f} "
            f"experiment={summary.experiment_name!r}"
        )


def main() -> None:
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
        help="Run against both anthropic and openai and report both (needs both API keys set).",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help=(
            "Also sync golden-dataset examples to LangSmith and run LLM-judge experiments "
            "(requires LANGSMITH_API_KEY)."
        ),
    )
    args = parser.parse_args()

    prereq_error = _langsmith_prereq_error(args.langsmith, os.environ.get("LANGSMITH_API_KEY"))
    if prereq_error is not None:
        print(prereq_error, file=sys.stderr)
        sys.exit(1)

    all_results: list[EvalResult] = []
    if args.compare:
        for provider in ("anthropic", "openai"):
            all_results += run_all(provider)  # type: ignore[arg-type]
    else:
        all_results += run_all(args.provider)  # type: ignore[arg-type]

    _print_report(all_results)
    output_path = save_results(all_results, _RESULTS_DIR)
    print(f"\nSaved results to {output_path}")

    langsmith_summaries: list[LangSmithEvalSummary] = []
    if args.langsmith:
        if args.compare:
            for provider in ("anthropic", "openai"):
                langsmith_summaries += run_langsmith_eval(provider)
        else:
            langsmith_summaries += run_langsmith_eval(args.provider)
        _print_langsmith_report(langsmith_summaries)

    if any(not result.passed for result in all_results) or any(
        summary.pass_rate < 1.0 for summary in langsmith_summaries
    ):
        print("\nRegression suite FAILED — one or more cases did not pass.")
        sys.exit(1)

    print("\nAll eval cases passed.")


if __name__ == "__main__":
    main()
