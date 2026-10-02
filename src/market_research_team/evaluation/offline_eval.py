"""Offline prompt evaluation: runs the golden dataset against a real LLM.

Unlike `tests/` (hermetic, fake LLMs, part of the free CI run), this makes
real API calls against whichever `LLM_PROVIDER` is configured (or an
explicit override) and checks structural/grounded properties of the
output — run it on demand (before a release, after changing a prompt or
switching models), not as part of the pytest suite.

Each `evaluate_*` function takes an `llm` directly rather than building
one internally, matching how `rewrite_and_expand`, `decide_next_step`,
`run_tool_calling_loop`, and `draft_report` are already structured
elsewhere — so the orchestration/scoring logic here is unit-testable with
a fake LLM, and only `run_all` (which calls `get_chat_model()`) needs
real credentials.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel

from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.agents.reporting.node import draft_report
from market_research_team.agents.research.node import run_research_pipeline
from market_research_team.agents.supervisor.router import decide_next_step
from market_research_team.config import settings
from market_research_team.evaluation import checks, judges
from market_research_team.evaluation.golden_dataset import (
    ANALYTICS_CASES,
    FULL_PIPELINE_CASES,
    QUERY_REWRITE_CASES,
    REPORTING_CASES,
    RETRIEVAL_CASES,
    SUPERVISOR_DECISION_CASES,
)
from market_research_team.evaluation.graph_runs import run_graph_with_decision
from market_research_team.evaluation.regression_eval import evaluate_regressions
from market_research_team.evaluation.results import EvalResult
from market_research_team.evaluation.safety_eval import evaluate_safety
from market_research_team.llm import get_chat_model
from market_research_team.retrieval.query_rewriter import rewrite_and_expand
from market_research_team.state import AgentState
from market_research_team.versioning import build_manifest, git_sha


def _safe_judge(
    judge_llm: BaseChatModel | None, judge: Callable[[], Any]
) -> tuple[float | None, str]:
    """Run a judge; returns (score, note to append to the case detail).

    No judge model -> (None, ""). A judge that raises or returns a
    non-numeric score leaves the case unjudged instead of failing it, and the
    error goes into the detail so a misconfigured judge is diagnosable --
    metrics.py reports the unjudged share and fails quality if it is too high.
    """

    if judge_llm is None:
        return None, ""
    try:
        score = float(judge().score)
    except Exception as exc:
        return None, f"; judge_error={type(exc).__name__}: {exc}"
    return min(max(score, 0.0), 1.0), ""


def evaluate_query_rewriter(
    llm: BaseChatModel,
    provider: str,
    *,
    judge_llm: BaseChatModel | None = None,
    repeat: int = 0,
) -> list[EvalResult]:
    results = []
    for case in QUERY_REWRITE_CASES:
        queries = rewrite_and_expand(case.objective, llm)
        count_passed, count_detail = checks.check_query_count(
            queries, min_count=case.min_queries, max_count=case.max_queries
        )
        keyword_passed, keyword_detail = checks.check_keyword_coverage(
            queries, case.required_any_keywords
        )
        passed = count_passed and keyword_passed
        detail = f"queries={queries!r}; {count_detail}; {keyword_detail}"
        score, judge_note = _safe_judge(
            judge_llm, lambda: judges.judge_query_rewrite(case.objective, queries, judge_llm)
        )
        detail += judge_note
        results.append(
            EvalResult("query_rewrite", case.name, passed, detail, provider, score, repeat)
        )
    return results


def evaluate_retrieval(provider: str, *, repeat: int = 0) -> list[EvalResult]:
    """Runs the real research pipeline (query rewrite -> retrieve -> rerank)
    against the seeded vector store and checks that the expected source
    documents made it through -- catches a retrieval/reranking regression
    that a query-rewrite check alone would miss. Needs a seeded vector
    store, same prerequisite as `evaluate_full_pipeline`.

    Uses `run_research_pipeline` directly (not the full graph) so this
    stays cheaper than `evaluate_full_pipeline`, skipping analytics/reporting.
    """

    results = []
    for case in RETRIEVAL_CASES:
        findings, _query_count, _candidate_count, _events = run_research_pipeline(case.objective)
        passed, detail = checks.check_source_coverage(
            findings, case.expected_sources, case.min_hits
        )
        results.append(EvalResult("retrieval", case.name, passed, detail, provider, repeat=repeat))
    return results


def evaluate_supervisor_decision(
    llm: BaseChatModel,
    provider: str,
    *,
    judge_llm: BaseChatModel | None = None,
    repeat: int = 0,
) -> list[EvalResult]:
    results = []
    for case in SUPERVISOR_DECISION_CASES:
        state: AgentState = {
            "messages": [],
            "objective": "Assess competitor pricing strategy",
            "next": "research",
            "research_findings": case.research_findings,
            "analytics_results": case.analytics_results,
            "report_path": case.report_path,
        }
        decision = decide_next_step(state, llm)
        passed = decision in case.allowed_decisions
        detail = f"decision={decision!r} (allowed: {case.allowed_decisions})"
        score, judge_note = _safe_judge(
            judge_llm,
            lambda: judges.judge_supervisor_decision(
                len(case.research_findings), len(case.analytics_results), decision, judge_llm
            ),
        )
        detail += judge_note
        results.append(
            EvalResult("supervisor_decision", case.name, passed, detail, provider, score, repeat)
        )
    return results


def evaluate_analytics(
    llm: BaseChatModel,
    provider: str,
    *,
    judge_llm: BaseChatModel | None = None,
    repeat: int = 0,
) -> list[EvalResult]:
    results = []
    for case in ANALYTICS_CASES:
        outputs = run_tool_calling_loop(llm, ANALYTICS_TOOLS, case.objective, case.findings)
        count_passed, count_detail = checks.check_min_length(
            outputs, case.min_tool_calls, "tool calls"
        )
        value_passed, value_detail = checks.check_any_value_matches(
            outputs, case.plausible_values, case.tolerance
        )
        passed = count_passed and value_passed
        detail = f"{count_detail}; {value_detail}"
        score, judge_note = _safe_judge(
            judge_llm, lambda: judges.judge_analytics(case.findings, outputs, judge_llm)
        )
        detail += judge_note
        results.append(EvalResult("analytics", case.name, passed, detail, provider, score, repeat))
        if case.expected_tools:
            called = sorted({output["metric"] for output in outputs})
            selected = any(tool in called for tool in case.expected_tools)
            results.append(
                EvalResult(
                    "tool_selection",
                    case.name,
                    selected,
                    f"called={called}; expected any of {case.expected_tools}",
                    provider,
                    repeat=repeat,
                )
            )
    return results


def evaluate_reporting(
    llm: BaseChatModel,
    provider: str,
    *,
    judge_llm: BaseChatModel | None = None,
    repeat: int = 0,
) -> list[EvalResult]:
    results = []
    for case in REPORTING_CASES:
        report = draft_report(case.objective, case.findings, case.results, llm)
        sections_passed, sections_detail = checks.check_contains_all(report, case.required_sections)
        facts_passed, facts_detail = checks.check_contains_all(report, case.required_facts)
        passed = sections_passed and facts_passed
        detail = f"{sections_detail}; {facts_detail}"
        score, judge_note = _safe_judge(
            judge_llm,
            lambda: judges.judge_report(
                case.objective, case.findings, case.results, report, judge_llm
            ),
        )
        detail += judge_note
        results.append(EvalResult("reporting", case.name, passed, detail, provider, score, repeat))
    return results


def evaluate_full_pipeline(
    provider: str, *, judge_llm: BaseChatModel | None = None, repeat: int = 0
) -> list[EvalResult]:
    """Runs the real compiled graph end to end (needs a seeded vector store),
    approving the report at the human-approval step so the run can finish."""

    results = []
    for case in FULL_PIPELINE_CASES:
        final_state = run_graph_with_decision(
            case.objective,
            thread_id=f"eval-{case.name}-r{repeat}",
            decision={"approved": True},
        )
        report_path = final_state.get("report_path")
        passed = final_state.get("error") is None and bool(report_path)
        detail = (
            f"error={final_state.get('error')!r}, "
            f"report_path={report_path!r}, "
            f"findings={len(final_state.get('research_findings', []))}, "
            f"results={len(final_state.get('analytics_results', []))}"
        )
        score, judge_note = None, ""
        if report_path and Path(report_path).exists():
            report_text = Path(report_path).read_text(encoding="utf-8")
            score, judge_note = _safe_judge(
                judge_llm,
                lambda: judges.judge_full_pipeline(case.objective, report_text, judge_llm),
            )
        detail += judge_note
        results.append(
            EvalResult("full_pipeline", case.name, passed, detail, provider, score, repeat)
        )
    return results


def run_all(
    provider: Literal["anthropic", "openai"] | None = None,
    *,
    repeat: int = 0,
    judge_llm: BaseChatModel | None = None,
) -> list[EvalResult]:
    """Run every eval category (safety included) against a real LLM.

    If `provider` is given, temporarily overrides `settings.llm_provider`
    for the duration of the run (restored afterwards even on error), which
    is what lets `scripts/run_evals.py --compare` run both providers
    back-to-back in one process. `judge_llm` scores quality; None skips
    judging (e.g. when LangSmith supplies quality instead).
    """

    original_provider = settings.llm_provider
    if provider is not None:
        settings.llm_provider = provider
    active_provider = settings.llm_provider

    try:
        llm = get_chat_model()
        judged = {"judge_llm": judge_llm, "repeat": repeat}
        results: list[EvalResult] = []
        results += evaluate_query_rewriter(llm, active_provider, **judged)
        results += evaluate_retrieval(active_provider, repeat=repeat)
        results += evaluate_supervisor_decision(llm, active_provider, **judged)
        results += evaluate_analytics(llm, active_provider, **judged)
        results += evaluate_reporting(llm, active_provider, **judged)
        results += evaluate_full_pipeline(active_provider, **judged)
        results += evaluate_safety(llm, active_provider, repeat=repeat)
        results += evaluate_regressions(active_provider, repeat=repeat)
        return results
    finally:
        settings.llm_provider = original_provider


@contextmanager
def scoped_eval_run(run_dir: Path) -> Iterator[Path]:
    """Point the audit log and report output at `run_dir` for one eval run, so
    latency/token/tool metrics cover only this run and eval reports don't land
    in `reports/`. Always restored, even when the run raises."""

    original_audit, original_reports = settings.audit_log_path, settings.reports_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    settings.audit_log_path = run_dir / "audit.jsonl"
    settings.reports_dir = run_dir / "reports"
    try:
        yield run_dir
    finally:
        settings.audit_log_path = original_audit
        settings.reports_dir = original_reports


def save_results(
    results: list[EvalResult],
    run_dir: Path,
    *,
    gate: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """`manifest` should be the evaluated provider's (the process-wide provider
    setting may differ, e.g. `--provider openai`); defaults to the current one."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.json"
    manifest = manifest or build_manifest()
    payload = {
        "agent_version": f"{manifest['release']}+{manifest['fingerprint']}",
        "git_sha": git_sha(),
        "manifest": manifest,
        "results": [asdict(result) for result in results],
        "gate": gate or [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
