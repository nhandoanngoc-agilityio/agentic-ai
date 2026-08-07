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
import sys
from pathlib import Path

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
    args = parser.parse_args()

    all_results: list[EvalResult] = []
    if args.compare:
        for provider in ("anthropic", "openai"):
            all_results += run_all(provider)  # type: ignore[arg-type]
    else:
        all_results += run_all(args.provider)  # type: ignore[arg-type]

    _print_report(all_results)
    output_path = save_results(all_results, _RESULTS_DIR)
    print(f"\nSaved results to {output_path}")

    if any(not result.passed for result in all_results):
        print("\nRegression suite FAILED — one or more cases did not pass.")
        sys.exit(1)

    print("\nAll eval cases passed.")


if __name__ == "__main__":
    main()
