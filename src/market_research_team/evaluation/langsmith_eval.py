"""LangSmith integration for the prompt evaluation regression suite: dataset sync and
category-tailored LLM-judge evaluators, layered on top of -- not replacing -- the deterministic
checks in evaluation/checks.py. Powers the --langsmith flag on scripts/run_evals.py.
"""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from market_research_team import graph as graph_module
from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.agents.reporting.node import draft_report
from market_research_team.agents.research.query_rewriter import rewrite_and_expand
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
from market_research_team.state import AgentState
from market_research_team.supervisor.router import decide_next_step


@dataclass
class JudgeScore:
    passed: bool
    score: float
    reasoning: str


@dataclass
class LangSmithEvalSummary:
    category: str
    # From ExperimentResults.experiment_name() -- the SDK doesn't hand back a ready-made UI
    # URL, so callers report the name (findable in the LangSmith UI) instead.
    experiment_name: str
    pass_rate: float


class _JudgeScoreSchema(BaseModel):
    """Structured-output schema for every llm_judge_* evaluator -- same shape as JudgeScore."""

    passed: bool = Field(description="Whether the output meets the judge's quality bar.")
    score: float = Field(description="A 0.0-1.0 quality score, 1.0 being the best.")
    reasoning: str = Field(description="Brief justification for the score.")


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


_QUERY_REWRITE_JUDGE_PROMPT = (
    "You are grading whether a set of search sub-queries are good, non-redundant "
    "decompositions of a research objective for a market and competitor research "
    "assistant. Score 1.0 if the queries clearly cover distinct angles of the "
    "objective, lower if they're redundant, off-topic, or too vague."
)


def llm_judge_query_rewrite(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    queries = (run.outputs or {}).get("queries", [])
    objective = (example.inputs or {}).get("objective", "")
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_QUERY_REWRITE_JUDGE_PROMPT),
            HumanMessage(content=f"Objective: {objective}\nQueries: {queries}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _supervisor_decision_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "research_findings": case.research_findings,
                "analytics_results": case.analytics_results,
                "report_path": case.report_path,
            },
            "outputs": {"allowed_decisions": list(case.allowed_decisions)},
        }
        for case in SUPERVISOR_DECISION_CASES
    ]


def target_supervisor_decision(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    state: AgentState = {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": inputs["research_findings"],
        "analytics_results": inputs["analytics_results"],
        "report_path": inputs["report_path"],
    }
    decision = decide_next_step(state, llm)
    return {"decision": decision}


def deterministic_evaluator_supervisor_decision(run: Any, example: Any) -> dict[str, Any]:
    decision = (run.outputs or {}).get("decision")
    allowed = (example.outputs or {}).get("allowed_decisions", [])
    passed = decision in allowed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"decision={decision!r} (allowed: {allowed})",
    }


_SUPERVISOR_JUDGE_PROMPT = (
    "You are grading whether a supervisor's routing decision for a market research "
    "multi-agent system is reasonable given the current state, not just technically "
    "allowed. Score 1.0 if the decision clearly makes sense given what's been "
    "gathered so far."
)


def llm_judge_supervisor_decision(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    decision = (run.outputs or {}).get("decision")
    findings_count = len((example.inputs or {}).get("research_findings", []))
    results_count = len((example.inputs or {}).get("analytics_results", []))
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_SUPERVISOR_JUDGE_PROMPT),
            HumanMessage(
                content=(
                    f"Research findings so far: {findings_count}\n"
                    f"Analytics results so far: {results_count}\n"
                    f"Decision made: {decision!r}"
                )
            ),
        ]
    )
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
    results = run_tool_calling_loop(llm, ANALYTICS_TOOLS, inputs["objective"], inputs["findings"])
    return {"results": results}


def deterministic_evaluator_analytics(run: Any, example: Any) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    expected = example.outputs or {}
    count_passed, count_detail = checks.check_min_length(
        results, expected.get("min_tool_calls", 1), "tool calls"
    )
    value_passed, value_detail = checks.check_any_value_matches(
        results, expected.get("plausible_values", []), expected.get("tolerance", 1.0)
    )
    passed = count_passed and value_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{count_detail}; {value_detail}",
    }


_ANALYTICS_JUDGE_PROMPT = (
    "You are grading whether computed analytics results are grounded in the given "
    "research findings for a market research system. Score 1.0 only if every "
    "reported value plausibly traces back to a number stated in the findings -- "
    "score low if any value looks invented."
)


def llm_judge_analytics(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    findings = (example.inputs or {}).get("findings", [])
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_ANALYTICS_JUDGE_PROMPT),
            HumanMessage(content=f"Findings: {findings}\nComputed results: {results}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _reporting_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "objective": case.objective,
                "findings": case.findings,
                "results": case.results,
            },
            "outputs": {
                "required_sections": case.required_sections,
                "required_facts": case.required_facts,
            },
        }
        for case in REPORTING_CASES
    ]


def target_reporting(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    report = draft_report(inputs["objective"], inputs["findings"], inputs["results"], llm)
    return {"report": report}


def deterministic_evaluator_reporting(run: Any, example: Any) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    expected = example.outputs or {}
    sections_passed, sections_detail = checks.check_contains_all(
        report, expected.get("required_sections", [])
    )
    required_facts = expected.get("required_facts", [])
    facts_passed, facts_detail = checks.check_contains_all(report, required_facts)
    passed = sections_passed and facts_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{sections_detail}; {facts_detail}",
    }


_REPORTING_JUDGE_PROMPT = (
    "You are grading a markdown research report for faithfulness to its given "
    "findings/results and overall structure/clarity. Score 1.0 only if the report "
    "is well-organized and introduces no facts beyond what was given."
)


def llm_judge_reporting(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    findings = (example.inputs or {}).get("findings", [])
    results = (example.inputs or {}).get("results", [])
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_REPORTING_JUDGE_PROMPT),
            HumanMessage(content=f"Findings: {findings}\nResults: {results}\nReport:\n{report}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}


def _full_pipeline_examples() -> list[dict[str, Any]]:
    return [
        {"inputs": {"objective": case.objective}, "outputs": {}} for case in FULL_PIPELINE_CASES
    ]


def target_full_pipeline(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    # llm accepted for signature uniformity with the other target_* functions (all bound via
    # functools.partial(fn, llm=llm) in run_langsmith_eval) -- run_graph builds its own LLMs
    # internally via get_chat_model(), so it's unused here.
    initial_state: AgentState = {
        "messages": [],
        "objective": inputs["objective"],
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }
    final_state = graph_module.run_graph(initial_state)
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


_FULL_PIPELINE_JUDGE_PROMPT = (
    "You are grading the overall quality of a market research report written by a "
    "multi-agent pipeline, given the original objective. Score 1.0 if the report "
    "substantively and coherently addresses the objective."
)


def llm_judge_full_pipeline(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    outputs = run.outputs or {}
    report_path = outputs.get("report_path")
    objective = (example.inputs or {}).get("objective", "")
    report_text = Path(report_path).read_text(encoding="utf-8") if report_path else ""
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_FULL_PIPELINE_JUDGE_PROMPT),
            HumanMessage(content=f"Objective: {objective}\nReport:\n{report_text}"),
        ]
    )
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


def run_langsmith_eval(
    provider: Literal["anthropic", "openai"] | None = None,
) -> list[LangSmithEvalSummary]:
    """Sync each category's dataset and run a LangSmith experiment against it, layering the
    existing deterministic checks with a category-tailored LLM-judge. Mirrors
    offline_eval.run_all()'s provider-override-then-restore pattern. Requires
    LANGSMITH_API_KEY to be set (validated by the caller -- see scripts/run_evals.py)."""

    original_provider = settings.llm_provider
    if provider is not None:
        settings.llm_provider = provider

    try:
        llm = get_chat_model()
        client = Client()
        summaries: list[LangSmithEvalSummary] = []
        for category, examples_fn, target_fn, deterministic_fn, judge_fn in _CATEGORY_SPECS:
            examples = examples_fn()
            dataset = sync_dataset(client, category, examples)
            results = evaluate(
                functools.partial(target_fn, llm=llm),
                data=dataset.name,
                evaluators=[deterministic_fn, functools.partial(judge_fn, llm=llm)],
                experiment_prefix=f"market-research-team-{category}",
                client=client,
            )
            rows = list(results)
            total = len(rows)
            fully_passed = sum(
                1
                for row in rows
                if all(result.score == 1.0 for result in row["evaluation_results"]["results"])
            )
            pass_rate = fully_passed / total if total else 0.0
            summaries.append(LangSmithEvalSummary(category, results.experiment_name(), pass_rate))
        return summaries
    finally:
        settings.llm_provider = original_provider
