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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel

from market_research_team import graph as graph_module
from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.agents.reporting.node import draft_report
from market_research_team.agents.supervisor.router import decide_next_step
from market_research_team.config import settings
from market_research_team.evaluation import checks
from market_research_team.evaluation.golden_dataset import (
    ANALYTICS_CASES,
    FULL_PIPELINE_CASES,
    QUERY_REWRITE_CASES,
    REPORTING_CASES,
    SUPERVISOR_DECISION_CASES,
)
from market_research_team.llm import get_chat_model
from market_research_team.retrieval.query_rewriter import rewrite_and_expand
from market_research_team.state import AgentState


@dataclass
class EvalResult:
    category: str
    case_name: str
    passed: bool
    detail: str
    provider: str


def evaluate_query_rewriter(llm: BaseChatModel, provider: str) -> list[EvalResult]:
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
        results.append(EvalResult("query_rewrite", case.name, passed, detail, provider))
    return results


def evaluate_supervisor_decision(llm: BaseChatModel, provider: str) -> list[EvalResult]:
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
        results.append(EvalResult("supervisor_decision", case.name, passed, detail, provider))
    return results


def evaluate_analytics(llm: BaseChatModel, provider: str) -> list[EvalResult]:
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
        results.append(EvalResult("analytics", case.name, passed, detail, provider))
    return results


def evaluate_reporting(llm: BaseChatModel, provider: str) -> list[EvalResult]:
    results = []
    for case in REPORTING_CASES:
        report = draft_report(case.objective, case.findings, case.results, llm)
        sections_passed, sections_detail = checks.check_contains_all(report, case.required_sections)
        facts_passed, facts_detail = checks.check_contains_all(report, case.required_facts)
        passed = sections_passed and facts_passed
        detail = f"{sections_detail}; {facts_detail}"
        results.append(EvalResult("reporting", case.name, passed, detail, provider))
    return results


def evaluate_full_pipeline(provider: str) -> list[EvalResult]:
    """Runs the real compiled graph end to end (needs a seeded vector store).

    Uses `graph_module.run_graph` (not a direct import) so tests can
    monkeypatch it, same pattern as the graph-level tests in
    `tests/agents/test_supervisor_routing.py`.
    """

    results = []
    for case in FULL_PIPELINE_CASES:
        initial_state: AgentState = {
            "messages": [],
            "objective": case.objective,
            "next": "research",
            "research_findings": [],
            "analytics_results": [],
            "report_path": None,
        }
        final_state = graph_module.run_graph(initial_state)
        passed = final_state.get("error") is None and bool(final_state.get("report_path"))
        detail = (
            f"error={final_state.get('error')!r}, "
            f"report_path={final_state.get('report_path')!r}, "
            f"findings={len(final_state.get('research_findings', []))}, "
            f"results={len(final_state.get('analytics_results', []))}"
        )
        results.append(EvalResult("full_pipeline", case.name, passed, detail, provider))
    return results


def run_all(provider: Literal["anthropic", "openai"] | None = None) -> list[EvalResult]:
    """Run every eval category against a real LLM.

    If `provider` is given, temporarily overrides `settings.llm_provider`
    for the duration of the run (restored afterwards even on error), which
    is what lets `scripts/run_evals.py --compare` run both providers
    back-to-back in one process.
    """

    original_provider = settings.llm_provider
    if provider is not None:
        settings.llm_provider = provider
    active_provider = settings.llm_provider

    try:
        llm = get_chat_model()
        results: list[EvalResult] = []
        results += evaluate_query_rewriter(llm, active_provider)
        results += evaluate_supervisor_decision(llm, active_provider)
        results += evaluate_analytics(llm, active_provider)
        results += evaluate_reporting(llm, active_provider)
        results += evaluate_full_pipeline(active_provider)
        return results
    finally:
        settings.llm_provider = original_provider


def save_results(results: list[EvalResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"eval_{timestamp}.json"
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    return path
