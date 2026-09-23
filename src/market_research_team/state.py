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
    # Set by `reporting_node` when a human explicitly discards a draft rather
    # than approving or rejecting it (the comparison UI's "losing" candidate).
    # A discard writes no report and records no error, so without this flag the
    # supervisor would see an unfinished run and route back to reporting
    # forever -- `decide_next_step` checks it to end the run instead.
    report_discarded: NotRequired[bool]
    # Append-only: every node that blocks, drops, redacts or flags something
    # adds an event here. Read by the audit record at FINISH and by the CLI.
    guardrail_events: NotRequired[Annotated[list[GuardrailEvent], operator.add]]
