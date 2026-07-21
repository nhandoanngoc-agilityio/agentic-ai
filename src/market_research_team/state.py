"""Unified state schema shared across the supervisor and every agent node."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

RouteDecision = Literal["research", "analytics", "reporting", "FINISH"]


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


class AgentState(TypedDict):
    """State threaded through the supervisor and every sub-agent node."""

    messages: Annotated[list[BaseMessage], add_messages]
    objective: str
    next: RouteDecision
    research_findings: list[ResearchFinding]
    analytics_results: list[AnalyticsResult]
    report_path: str | None
