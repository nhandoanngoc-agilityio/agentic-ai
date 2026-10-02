"""Curated production failures (evals/regressions.jsonl) as release-gate cases.

Pass/fail only, no LLM judge: `must_block` cases run just the input guard
(free); the rest run the real graph once per repeat, approving at the approval
step, and check the curator's declarative expectations.
"""

from pathlib import Path

from market_research_team.evaluation.golden_dataset import REGRESSION_CASES
from market_research_team.evaluation.graph_runs import GraphRunOutcome, run_graph_with_outcome
from market_research_team.evaluation.results import EvalResult
from market_research_team.feedback.regressions import RegressionEntry
from market_research_team.security.input_guard import input_guard_node


def _check_blocked(case: RegressionEntry) -> tuple[bool, str]:
    update = input_guard_node({"objective": case.objective})  # type: ignore[typeddict-item]
    blocked = bool(update.get("error")) and any(
        event["layer"] == "input" for event in update.get("guardrail_events", [])
    )
    return blocked, f"blocked={blocked}"


def _check_graph(case: RegressionEntry, outcome: GraphRunOutcome) -> tuple[bool, str]:
    e = case.expectations
    state = outcome.state
    report_path = state.get("report_path")
    text = ""
    if report_path and Path(report_path).exists():
        text = Path(report_path).read_text(encoding="utf-8")
    lowered = text.lower()
    problems: list[str] = []
    if state.get("error"):
        problems.append(f"error={state.get('error')!r}")
    if (e.required_facts or e.forbidden_substrings) and not text:
        problems.append("no report written")
    missing = [f for f in e.required_facts if f.lower() not in lowered]
    if missing:
        problems.append(f"missing facts {missing}")
    present = [s for s in e.forbidden_substrings if s.lower() in lowered]
    if present:
        problems.append(f"forbidden text present {present}")
    findings = len(state.get("research_findings", []))
    if e.min_findings is not None and findings < e.min_findings:
        problems.append(f"findings {findings} < {e.min_findings}")
    if e.requires_approval and (outcome.approval_rounds < 1 or outcome.wrote_before_approval):
        problems.append(
            f"did not ask for approval before writing (rounds={outcome.approval_rounds})"
        )
    return not problems, "; ".join(problems) or "all expectations met"


def evaluate_regressions(
    provider: str, *, repeat: int = 0, cases: list[RegressionEntry] | None = None
) -> list[EvalResult]:
    results = []
    for case in REGRESSION_CASES if cases is None else cases:
        try:
            if case.expectations.must_block:
                passed, detail = _check_blocked(case)
            else:
                outcome = run_graph_with_outcome(
                    case.objective,
                    thread_id=f"eval-regression-{case.id}-r{repeat}",
                    decision={"approved": True},
                )
                passed, detail = _check_graph(case, outcome)
        except Exception as exc:  # a crashing case is a failing case
            passed, detail = False, f"check raised {type(exc).__name__}: {exc}"
        results.append(EvalResult("regression", case.id, passed, detail, provider, repeat=repeat))
    return results
