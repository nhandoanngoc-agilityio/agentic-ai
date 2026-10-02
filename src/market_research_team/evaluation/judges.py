"""LLM-judge prompts and plain-input judge functions.

Shared by the offline release gate (`offline_eval.py`) and the LangSmith
evaluators (`langsmith_eval.py`) so each judge has exactly one prompt.
`judge_prompts_hash()` is stored in the gate baseline: when it changes,
quality scores are no longer like-for-like and the gate warns.
"""

import hashlib
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class JudgeScoreSchema(BaseModel):
    """Structured-output schema every judge returns."""

    passed: bool = Field(description="Whether the output meets the judge's quality bar.")
    score: float = Field(description="A 0.0-1.0 quality score, 1.0 being the best.")
    reasoning: str = Field(description="Brief justification for the score.")


QUERY_REWRITE_JUDGE_PROMPT = (
    "You are grading whether a set of search sub-queries are good, non-redundant "
    "decompositions of a research objective for a market and competitor research "
    "assistant. Score 1.0 if the queries clearly cover distinct angles of the "
    "objective, lower if they're redundant, off-topic, or too vague."
)
SUPERVISOR_JUDGE_PROMPT = (
    "You are grading whether a supervisor's routing decision for a market research "
    "multi-agent system is reasonable given the current state, not just technically "
    "allowed. Score 1.0 if the decision clearly makes sense given what's been "
    "gathered so far."
)
ANALYTICS_JUDGE_PROMPT = (
    "You are grading whether computed analytics results are grounded in the given "
    "research findings for a market research system. Score 1.0 only if every "
    "reported value plausibly traces back to a number stated in the findings -- "
    "score low if any value looks invented."
)
REPORTING_JUDGE_PROMPT = (
    "You are grading a markdown research report for faithfulness to its given "
    "findings/results and overall structure/clarity. Score 1.0 only if the report "
    "is well-organized and introduces no facts beyond what was given."
)
FULL_PIPELINE_JUDGE_PROMPT = (
    "You are grading the overall quality of a market research report written by a "
    "multi-agent pipeline, given the original objective. Score 1.0 if the report "
    "substantively and coherently addresses the objective."
)

_ALL_PROMPTS = (
    QUERY_REWRITE_JUDGE_PROMPT,
    SUPERVISOR_JUDGE_PROMPT,
    ANALYTICS_JUDGE_PROMPT,
    REPORTING_JUDGE_PROMPT,
    FULL_PIPELINE_JUDGE_PROMPT,
)


def _judge(llm: BaseChatModel, system_prompt: str, human: str) -> JudgeScoreSchema:
    structured_llm = llm.with_structured_output(JudgeScoreSchema)
    return structured_llm.invoke(  # type: ignore[return-value]
        [SystemMessage(content=system_prompt), HumanMessage(content=human)]
    )


def judge_query_rewrite(objective: str, queries: list[str], llm: BaseChatModel) -> JudgeScoreSchema:
    return _judge(llm, QUERY_REWRITE_JUDGE_PROMPT, f"Objective: {objective}\nQueries: {queries}")


def judge_supervisor_decision(
    findings_count: int, results_count: int, decision: str | None, llm: BaseChatModel
) -> JudgeScoreSchema:
    return _judge(
        llm,
        SUPERVISOR_JUDGE_PROMPT,
        (
            f"Research findings so far: {findings_count}\n"
            f"Analytics results so far: {results_count}\n"
            f"Decision made: {decision!r}"
        ),
    )


def judge_analytics(
    findings: list[Any], results: list[Any], llm: BaseChatModel
) -> JudgeScoreSchema:
    return _judge(llm, ANALYTICS_JUDGE_PROMPT, f"Findings: {findings}\nComputed results: {results}")


def judge_report(
    objective: str, findings: list[Any], results: list[Any], report: str, llm: BaseChatModel
) -> JudgeScoreSchema:
    return _judge(
        llm,
        REPORTING_JUDGE_PROMPT,
        f"Objective: {objective}\nFindings: {findings}\nResults: {results}\nReport:\n{report}",
    )


def judge_full_pipeline(objective: str, report_text: str, llm: BaseChatModel) -> JudgeScoreSchema:
    return _judge(
        llm, FULL_PIPELINE_JUDGE_PROMPT, f"Objective: {objective}\nReport:\n{report_text}"
    )


def judge_prompts_hash() -> str:
    return hashlib.sha256("\n".join(_ALL_PROMPTS).encode("utf-8")).hexdigest()[:12]


def judge_id(source: str, provider: str, model: str) -> str:
    """Identifies what produced quality scores: where they were judged (local
    or LangSmith), by which model, with which prompts. Stored in the gate
    baseline; a different id means quality is not like-for-like."""

    return f"{source}:{provider}/{model}:{judge_prompts_hash()}"
