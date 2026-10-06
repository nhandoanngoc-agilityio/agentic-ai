"""Unified state schema shared across the supervisor and every agent node."""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

RouteDecision = Literal["research", "analytics", "reporting", "FINISH"]

GuardrailLayer = Literal["input", "retrieval", "tool", "output", "policy"]


class GuardrailEvent(TypedDict):
    """One guardrail decision, recorded so the audit log and the reviewer can see
    what was blocked, dropped, redacted or flagged during a run."""

    layer: GuardrailLayer
    rule: str
    detail: str


class ResearchFinding(TypedDict):
    """A single retrieved-and-reranked piece of evidence."""

    source: str
    content: str
    relevance_score: float


class AnalyticsResult(TypedDict):
    """A single computed metric surfaced by the Analytics Agent."""

    metric: str
    value: float
    detail: str
    # Which compared subject this metric is about (e.g. "Acme"), when the
    # objective compares multiple named entities. None for single-subject
    # runs; set from the `entity` tool argument in `run_tool_calling_loop`.
    entity: str | None


class AgentState(TypedDict):
    """State threaded through the supervisor and every sub-agent node."""

    messages: Annotated[list[BaseMessage], add_messages]
    objective: str
    next: RouteDecision
    research_findings: list[ResearchFinding]
    analytics_results: list[AnalyticsResult]
    report_path: str | None
    error: NotRequired[str | None]
    # Set by `report_review_node` when a human explicitly discards a draft rather
    # than approving or rejecting it (the comparison UI's "losing" candidate).
    # A discard writes no report and records no error, so without this flag the
    # supervisor would see an unfinished run and route back to reporting
    # forever -- `decide_next_step` checks it to end the run instead.
    report_discarded: NotRequired[bool]
    # Targeted re-research. When the supervisor hands work back to Research it
    # names the gap (`research_focus`); Research rewrites queries for that gap
    # and merges new findings into the existing ones. A pass that adds nothing
    # new sets `research_exhausted`, after which Research is no longer offered.
    research_focus: NotRequired[str | None]
    research_exhausted: NotRequired[bool]
    # Hand-off between `reporting_node` (drafts) and `report_review_node`
    # (interrupt + write). The draft lives in checkpointed state so resuming
    # the review never re-drafts: the file written is byte-for-byte the draft
    # the human approved. `report_review_round` is 1-based once drafting starts
    # and reset to 0 when the review concludes; `report_feedback` is the last
    # rejection reason, folded into the next draft.
    report_draft: NotRequired[str | None]
    report_draft_warnings: NotRequired[list[str]]
    report_review_round: NotRequired[int]
    report_feedback: NotRequired[str | None]
    # Append-only: every node that blocks, drops, redacts or flags something
    # adds an event here. Read by the audit record at FINISH and by the CLI.
    guardrail_events: NotRequired[Annotated[list[GuardrailEvent], operator.add]]
    # Set by `run_graph` when `research_findings`/`analytics_results` were
    # populated from the response cache (see caching/response_cache.py)
    # rather than computed fresh. `decide_next_step` uses this to skip
    # straight to `reporting` -- drafting and human approval still always
    # run fresh regardless of this flag.
    from_response_cache: NotRequired[bool]
