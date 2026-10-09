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


PlanStatus = Literal["open", "answered", "unanswerable"]


class PlanItem(TypedDict):
    """One sub-question of the objective, tracked until it is answered.

    The planner writes the items (all `open`). The supervisor marks an item
    `answered` once findings from named sources cover it -- code checks those
    sources exist. Research marks an item `attempted` after a pass targeted
    at it, and `unanswerable` when that pass finds nothing new: the knowledge
    base doesn't hold the answer.
    """

    id: str
    question: str
    status: PlanStatus
    sources: list[str]
    # Set by Research after a pass targeted at this item. Each item gets one
    # targeted search: retargeting the same question mostly re-retrieves the
    # same chunks, so a second pass costs a loop iteration for little gain.
    attempted: NotRequired[bool]


class ResearchFinding(TypedDict):
    """A single retrieved-and-reranked piece of evidence."""

    source: str
    content: str
    relevance_score: float
    # Provenance from the document header and the chunk's headings (see
    # retrieval/evidence.py). Absent for undated documents.
    entity: NotRequired[str]
    topic: NotRequired[str]
    as_of: NotRequired[str]
    doc_type: NotRequired[str]


class AnalyticsResult(TypedDict):
    """A single computed metric surfaced by the Analytics Agent."""

    metric: str
    value: float
    detail: str
    # Which compared subject this metric is about (e.g. "Acme"), when the
    # objective compares multiple named entities. None for single-subject
    # runs; set from the `entity` tool argument in `run_tool_calling_loop`.
    entity: str | None
    # The numeric arguments the model passed to the tool. Checked against the
    # findings (`output_filters.ungrounded_inputs`): an input the findings
    # don't contain is a figure the model supplied, not one it read.
    inputs: NotRequired[list[float]]
    # What the metric means, in the model's words (e.g. "Globex ACV range"),
    # from the tool's `label` argument. `metric` stays the tool name: the UI
    # chart groups results by it across entities.
    label: NotRequired[str | None]
    # The result's ID within one analytics pass ("r1", "r2", ... in call
    # order), so insights can cite the computations they rest on.
    id: NotRequired[str]


class AnalyticsInsight(TypedDict):
    """One claim Analytics draws from its computations, for Reporting.

    Validated before it reaches state (security/output_filters.py::
    validate_insights): it cites real result IDs and states no figure the
    evidence doesn't support."""

    text: str
    result_ids: list[str]


class AgentState(TypedDict):
    """State threaded through the supervisor and every sub-agent node."""

    messages: Annotated[list[BaseMessage], add_messages]
    objective: str
    next: RouteDecision
    research_findings: list[ResearchFinding]
    analytics_results: list[AnalyticsResult]
    # Analytics' validated summary of its results; replaced on each pass.
    # Empty when the model didn't submit one (Reporting then renders the raw
    # results as before).
    analytics_insights: NotRequired[list[AnalyticsInsight]]
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
    # The plan: the objective decomposed into sub-questions (`planner` node),
    # whose coverage drives the supervisor. `research_focus_id` names the plan
    # item a hand-back targets, so Research can mark it unanswerable if the
    # targeted pass adds nothing. Without a plan (empty list) routing falls
    # back to `research_exhausted` alone.
    plan: NotRequired[list[PlanItem]]
    research_focus_id: NotRequired[str | None]
    # Research bumps `findings_version` whenever it adds findings; Analytics
    # records the version it analyzed. Analytics is offered again only when
    # the findings changed since, so a re-run always has something new.
    findings_version: NotRequired[int]
    analyzed_findings_version: NotRequired[int]
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
    # The step whose exception ended the run (set by the error boundary), so
    # `graph.retry_failed_run` can re-run just that step from the checkpoint.
    failed_node: NotRequired[str | None]
    # How many times the supervisor has decided so far in this run. The visit
    # cap reads this rather than counting `messages`, which is a log only.
    supervisor_visits: NotRequired[int]


def new_run_state(objective: str) -> AgentState:
    """The input for a fresh run, with every per-run field set explicitly.

    The one builder the CLI, the Gradio app and the eval harness share, so a
    field added to `AgentState` is reset in exactly one place.
    """

    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "analytics_insights": [],
        "report_path": None,
        "error": None,
        "failed_node": None,
        "report_discarded": False,
        "research_focus": None,
        "research_exhausted": False,
        "plan": [],
        "research_focus_id": None,
        "findings_version": 0,
        "analyzed_findings_version": 0,
        "report_draft": None,
        "report_draft_warnings": [],
        "report_review_round": 0,
        "report_feedback": None,
        "guardrail_events": [],
        "from_response_cache": False,
        "supervisor_visits": 0,
    }
