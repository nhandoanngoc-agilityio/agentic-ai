"""LangSmith integration for the prompt evaluation regression suite: dataset sync and
category-tailored LLM-judge evaluators, layered on top of -- not replacing -- the deterministic
checks in evaluation/checks.py. Powers the --langsmith flag on scripts/run_evals.py.
"""

import functools
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langsmith import Client, evaluate

from market_research_team.agents.analytics.node import run_analysis_loop
from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.agents.reporting.node import draft_report
from market_research_team.agents.supervisor.router import decide_route
from market_research_team.config import settings
from market_research_team.evaluation import checks, judges
from market_research_team.evaluation.golden_dataset import (
    ANALYTICS_CASES,
    FULL_PIPELINE_CASES,
    QUERY_REWRITE_CASES,
    REPORTING_CASES,
    SUPERVISOR_DECISION_CASES,
)
from market_research_team.evaluation.graph_runs import run_graph_with_decision
from market_research_team.evaluation.judges import (
    JudgeScoreSchema as _JudgeScoreSchema,  # noqa: F401
)
from market_research_team.evaluation.judges import judge_report
from market_research_team.llm import get_chat_model
from market_research_team.retrieval.query_rewriter import rewrite_and_expand
from market_research_team.security.output_filters import validate_insights
from market_research_team.state import AgentState
from market_research_team.versioning import trace_metadata


@dataclass
class JudgeScore:
    passed: bool
    score: float
    reasoning: str


@dataclass
class LangSmithEvalSummary:
    category: str
    # From ExperimentResults.experiment_name -- the SDK doesn't hand back a ready-made UI
    # URL, so callers report the name (findable in the LangSmith UI) instead.
    experiment_name: str
    pass_rate: float
    # Mean of the `llm_judge` scores across the experiment's rows; the gate
    # uses it as the quality metric when --langsmith is passed.
    mean_judge_score: float | None = None
    judged_count: int = 0
    # Rows in the experiment; rows minus judged_count were not judged (judge error).
    # None means unknown and is treated as judged_count.
    row_count: int | None = None


def sync_dataset(client: Any, category: str, examples: list[dict[str, Any]]) -> Any:
    """Get-or-create `market-research-team-{category}`, wipe its examples, recreate from the
    current in-repo cases. LangSmith is always a mirror of golden_dataset.py, never a second
    source of truth -- this makes every --langsmith run reflect the latest local cases with no
    drift or manual dataset editing in the LangSmith UI."""

    dataset_name = f"market-research-team-{category}"
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        for existing in client.list_examples(dataset_name=dataset_name):
            client.delete_example(example_id=existing.id)
    else:
        dataset = client.create_dataset(
            dataset_name, description=f"Golden dataset mirror: {category}"
        )

    for example in examples:
        client.create_example(
            inputs=example["inputs"], outputs=example["outputs"], dataset_id=dataset.id
        )
    return dataset


def _query_rewrite_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"objective": case.objective},
            "outputs": {
                "min_queries": case.min_queries,
                "max_queries": case.max_queries,
                "required_any_keywords": case.required_any_keywords,
            },
        }
        for case in QUERY_REWRITE_CASES
    ]


def target_query_rewrite(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    queries = rewrite_and_expand(inputs["objective"], llm)
    return {"queries": queries}


def deterministic_evaluator_query_rewrite(run: Any, example: Any) -> dict[str, Any]:
    queries = (run.outputs or {}).get("queries", [])
    expected = example.outputs or {}
    count_passed, count_detail = checks.check_query_count(
        queries,
        min_count=expected.get("min_queries", 1),
        max_count=expected.get("max_queries", 6),
    )
    keyword_passed, keyword_detail = checks.check_keyword_coverage(
        queries, expected.get("required_any_keywords", [])
    )
    passed = count_passed and keyword_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{count_detail}; {keyword_detail}",
    }


def llm_judge_query_rewrite(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    queries = (run.outputs or {}).get("queries", [])
    objective = (example.inputs or {}).get("objective", "")
    judgment = judges.judge_query_rewrite(objective, queries, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _supervisor_decision_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "objective": case.objective,
                "plan": case.plan,
                "research_findings": case.research_findings,
                "analytics_results": case.analytics_results,
                "report_path": case.report_path,
            },
            "outputs": {
                "allowed_decisions": list(case.allowed_decisions),
                "focus_keywords": case.focus_keywords,
            },
        }
        for case in SUPERVISOR_DECISION_CASES
    ]


def target_supervisor_decision(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    state: AgentState = {
        "messages": [],
        "objective": inputs.get("objective", "Assess competitor pricing strategy"),
        "next": "research",
        "research_findings": inputs["research_findings"],
        "analytics_results": inputs["analytics_results"],
        "report_path": inputs["report_path"],
        "plan": [dict(item) for item in inputs.get("plan", [])],  # type: ignore[misc]
    }
    route = decide_route(state, llm)
    return {
        "decision": route.next,
        "focus": route.research_focus,
        "rationale": route.rationale,
        "decided_by": route.decided_by,
    }


def deterministic_evaluator_supervisor_decision(run: Any, example: Any) -> dict[str, Any]:
    outputs = run.outputs or {}
    expected = example.outputs or {}
    passed, detail = checks.check_supervisor_route(
        outputs.get("decision"),  # type: ignore[arg-type]
        outputs.get("focus"),
        tuple(expected.get("allowed_decisions", [])),
        expected.get("focus_keywords", []),
        outputs.get("decided_by", "llm"),
    )
    return {"key": "deterministic", "score": 1.0 if passed else 0.0, "comment": detail}


def llm_judge_supervisor_decision(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    decision = (run.outputs or {}).get("decision")
    findings_count = len((example.inputs or {}).get("research_findings", []))
    results_count = len((example.inputs or {}).get("analytics_results", []))
    judgment = judges.judge_supervisor_decision(findings_count, results_count, decision, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _analytics_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"objective": case.objective, "findings": case.findings},
            "outputs": {
                "min_tool_calls": case.min_tool_calls,
                "plausible_values": case.plausible_values,
                "tolerance": case.tolerance,
            },
        }
        for case in ANALYTICS_CASES
    ]


def target_analytics(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    results, raw = run_analysis_loop(llm, ANALYTICS_TOOLS, inputs["objective"], inputs["findings"])
    insights, events = validate_insights(raw, results, inputs["findings"])
    return {"results": results, "insights": insights, "insight_events": events}


def deterministic_evaluator_analytics(run: Any, example: Any) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    expected = example.outputs or {}
    count_passed, count_detail = checks.check_min_length(
        results, expected.get("min_tool_calls", 1), "tool calls"
    )
    value_passed, value_detail = checks.check_any_value_matches(
        results, expected.get("plausible_values", []), expected.get("tolerance", 1.0)
    )
    inputs_passed, inputs_detail = checks.check_inputs_grounded(
        results, (example.inputs or {}).get("findings", [])
    )
    current_passed, current_detail = checks.check_inputs_current(
        results, (example.inputs or {}).get("findings", [])
    )
    outputs = run.outputs or {}
    insight_passed, insight_detail = checks.check_insights(
        outputs.get("insights", []), outputs.get("insight_events", [])
    )
    passed = count_passed and value_passed and inputs_passed and current_passed and insight_passed
    comment = f"{count_detail}; {value_detail}; {inputs_detail}; {current_detail}; {insight_detail}"
    return {"key": "deterministic", "score": 1.0 if passed else 0.0, "comment": comment}


def llm_judge_analytics(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    findings = (example.inputs or {}).get("findings", [])
    judgment = judges.judge_analytics(findings, results, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _reporting_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "objective": case.objective,
                "findings": case.findings,
                "results": case.results,
                "feedback": case.feedback,
                "insights": case.insights,
            },
            "outputs": {
                "required_sections": case.required_sections,
                "required_facts": case.required_facts,
                "require_table": case.require_table,
            },
        }
        for case in REPORTING_CASES
    ]


def target_reporting(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    report = draft_report(
        inputs["objective"],
        inputs["findings"],
        inputs["results"],
        llm,
        feedback=inputs.get("feedback"),
        insights=inputs.get("insights") or None,
    )
    return {"report": report}


def deterministic_evaluator_reporting(run: Any, example: Any) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    expected = example.outputs or {}
    sections_passed, sections_detail = checks.check_contains_all(
        report, expected.get("required_sections", [])
    )
    required_facts = expected.get("required_facts", [])
    facts_passed, facts_detail = checks.check_contains_all(report, required_facts)
    passed, comment = sections_passed and facts_passed, f"{sections_detail}; {facts_detail}"
    fresh_passed, fresh_detail = checks.check_freshness(
        report, (example.inputs or {}).get("findings", [])
    )
    passed, comment = passed and fresh_passed, f"{comment}; {fresh_detail}"
    if expected.get("require_table"):
        table_passed, table_detail = checks.check_markdown_table(report)
        passed, comment = passed and table_passed, f"{comment}; {table_detail}"
    return {"key": "deterministic", "score": 1.0 if passed else 0.0, "comment": comment}


def llm_judge_reporting(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    objective = (example.inputs or {}).get("objective", "")
    findings = (example.inputs or {}).get("findings", [])
    results = (example.inputs or {}).get("results", [])
    judgment = judge_report(objective, findings, results, report, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _full_pipeline_examples() -> list[dict[str, Any]]:
    return [
        {"inputs": {"objective": case.objective}, "outputs": {}} for case in FULL_PIPELINE_CASES
    ]


def target_full_pipeline(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    # llm accepted for signature uniformity with the other target_* functions (all bound via
    # functools.partial(fn, llm=llm) in run_langsmith_eval) -- the graph builds its own LLMs
    # internally via get_chat_model(), so it's unused here. The approval interrupt is
    # answered with an approval, same as the offline full-pipeline eval.
    final_state = run_graph_with_decision(
        inputs["objective"],
        thread_id=f"eval-langsmith-{uuid.uuid4().hex[:8]}",
        decision={"approved": True},
    )
    return {
        "error": final_state.get("error"),
        "report_path": final_state.get("report_path"),
        "findings_count": len(final_state.get("research_findings", [])),
        "results_count": len(final_state.get("analytics_results", [])),
    }


def deterministic_evaluator_full_pipeline(run: Any, example: Any) -> dict[str, Any]:
    outputs = run.outputs or {}
    passed = outputs.get("error") is None and bool(outputs.get("report_path"))
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"error={outputs.get('error')!r}, report_path={outputs.get('report_path')!r}",
    }


def llm_judge_full_pipeline(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    outputs = run.outputs or {}
    report_path = outputs.get("report_path")
    objective = (example.inputs or {}).get("objective", "")
    report_text = Path(report_path).read_text(encoding="utf-8") if report_path else ""
    judgment = judges.judge_full_pipeline(objective, report_text, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


_CATEGORY_SPECS: list[tuple[str, Any, Any, Any, Any]] = [
    (
        "query_rewrite",
        _query_rewrite_examples,
        target_query_rewrite,
        deterministic_evaluator_query_rewrite,
        llm_judge_query_rewrite,
    ),
    (
        "supervisor_decision",
        _supervisor_decision_examples,
        target_supervisor_decision,
        deterministic_evaluator_supervisor_decision,
        llm_judge_supervisor_decision,
    ),
    (
        "analytics",
        _analytics_examples,
        target_analytics,
        deterministic_evaluator_analytics,
        llm_judge_analytics,
    ),
    (
        "reporting",
        _reporting_examples,
        target_reporting,
        deterministic_evaluator_reporting,
        llm_judge_reporting,
    ),
    (
        "full_pipeline",
        _full_pipeline_examples,
        target_full_pipeline,
        deterministic_evaluator_full_pipeline,
        llm_judge_full_pipeline,
    ),
]


def _experiment_name(results: Any) -> str:
    """`ExperimentResults.experiment_name` is a property in current langsmith
    SDKs (0.14+) and was a method in older ones; pyproject allows both."""

    name = results.experiment_name
    return str(name() if callable(name) else name)


def langsmith_api_key() -> str | None:
    """The LangSmith key from the shell environment, else from `.env` (via settings)."""

    return (
        os.environ.get("LANGSMITH_API_KEY")
        or settings.langsmith_api_key
        or settings.langchain_api_key
    )


def run_langsmith_eval(
    provider: Literal["anthropic", "openai"] | None = None,
    *,
    judge_llm: BaseChatModel | None = None,
    repeats: int = 1,
) -> list[LangSmithEvalSummary]:
    """Sync each category's dataset and run a LangSmith experiment against it, layering the
    existing deterministic checks with a category-tailored LLM-judge. Mirrors
    offline_eval.run_all()'s provider-override-then-restore pattern. Requires
    LANGSMITH_API_KEY to be set (validated by the caller -- see scripts/run_evals.py).
    `judge_llm` grades the outputs (the release gate pins it so the candidate never grades
    itself); without it the candidate model judges, as before. `repeats` runs every example
    that many times (LangSmith `num_repetitions`), so `run_evals.py --repeats 3` averages
    the judged quality over three samples instead of one -- with one judged row for some
    categories, a single judge call otherwise swings the gate."""

    original_provider = settings.llm_provider
    if provider is not None:
        settings.llm_provider = provider

    try:
        llm = get_chat_model()
        judge = judge_llm or llm
        # The LangSmith client reads its key from os.environ only; a key kept in
        # `.env` reaches it this way (never overriding one already exported).
        key = langsmith_api_key()
        if key:
            os.environ.setdefault("LANGSMITH_API_KEY", key)
        client = Client()
        summaries: list[LangSmithEvalSummary] = []
        for category, examples_fn, target_fn, deterministic_fn, judge_fn in _CATEGORY_SPECS:
            examples = examples_fn()
            dataset = sync_dataset(client, category, examples)
            results = evaluate(
                functools.partial(target_fn, llm=llm),
                data=dataset.name,
                evaluators=[deterministic_fn, functools.partial(judge_fn, llm=judge)],
                experiment_prefix=f"market-research-team-{category}",
                metadata=trace_metadata(),
                client=client,
                num_repetitions=repeats,
            )
            rows = list(results)
            total = len(rows)
            fully_passed = sum(
                1
                for row in rows
                if all(result.score == 1.0 for result in row["evaluation_results"]["results"])
            )
            pass_rate = fully_passed / total if total else 0.0
            judge_scores = [
                float(result.score)
                for row in rows
                for result in row["evaluation_results"]["results"]
                if getattr(result, "key", None) == "llm_judge" and result.score is not None
            ]
            mean_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None
            summaries.append(
                LangSmithEvalSummary(
                    category,
                    _experiment_name(results),
                    pass_rate,
                    mean_judge_score=mean_judge,
                    judged_count=len(judge_scores),
                    row_count=total,
                )
            )
        return summaries
    finally:
        settings.llm_provider = original_provider
