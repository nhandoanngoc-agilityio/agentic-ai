"""Pull production failure signals from Langfuse into RunEvidence.

Two signals, merged with the audit log by thread_id (== Langfuse session_id):
`user_feedback` scores of 0 (thumbs down, see
observability.record_feedback_score) and ERROR-level observations. Any API
failure returns no evidence plus a warning, so a harvest always falls back to
the audit log.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from market_research_team.feedback.signals import EVAL_THREAD_PREFIX, RunEvidence

MAX_ITEMS = 1000
_PAGE_SIZE = 100


def _iso(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


class _Collector:
    def __init__(self, api: Any) -> None:
        self.api = api
        self.evidence: dict[str, RunEvidence] = {}
        self.traces: dict[str, Any] = {}
        self.warnings: list[str] = []

    def trace(self, trace_id: str) -> Any:
        if trace_id not in self.traces:
            self.traces[trace_id] = self.api.trace.get(trace_id)
        return self.traces[trace_id]

    def safe_trace(self, trace_id: str) -> Any | None:
        """One missing or failing trace skips that item, not the whole harvest."""

        try:
            return self.trace(trace_id)
        except Exception as exc:
            self.warnings.append(f"Langfuse trace {trace_id} skipped ({type(exc).__name__}).")
            return None

    def run_for(self, session_id: str | None, trace_id: str | None) -> RunEvidence | None:
        if not session_id and trace_id:
            session_id = getattr(self.safe_trace(trace_id), "session_id", None)
        if not session_id or str(session_id).startswith(EVAL_THREAD_PREFIX):
            return None
        ev = self.evidence.setdefault(session_id, RunEvidence(session_id, sources=["langfuse"]))
        if trace_id and trace_id not in ev.langfuse_trace_ids:
            ev.langfuse_trace_ids.append(trace_id)
        return ev

    def scores(self, since: datetime) -> None:
        page, seen = 1, 0
        while True:
            response = self.api.scores.get_many(
                name="user_feedback",
                value=0,
                operator="=",
                from_timestamp=since,
                page=page,
                limit=_PAGE_SIZE,
            )
            for score in response.data:
                if seen >= MAX_ITEMS:
                    self.warnings.append(f"Langfuse scores: stopped at the {MAX_ITEMS}-item cap.")
                    return
                seen += 1
                ev = self.run_for(getattr(score, "session_id", None), score.trace_id)
                if ev is not None:
                    ev.ratings.append("down")
                    ev.seen(_iso(getattr(score, "timestamp", None)))
            if page >= getattr(response.meta, "total_pages", page):
                return
            page += 1

    def errors(self, since: datetime) -> None:
        cursor, seen = None, 0
        while True:
            response = self.api.observations.get_many(
                level="ERROR", from_start_time=since, limit=_PAGE_SIZE, cursor=cursor
            )
            for obs in response.data:
                if seen >= MAX_ITEMS:
                    self.warnings.append(f"Langfuse errors: stopped at the {MAX_ITEMS}-item cap.")
                    return
                seen += 1
                ev = self.run_for(getattr(obs, "session_id", None), obs.trace_id)
                if ev is not None:
                    ev.langfuse_errors.append(obs.status_message or obs.name or "error")
                    ev.seen(_iso(getattr(obs, "start_time", None)))
            cursor = getattr(response.meta, "cursor", None)
            if not cursor:
                return

    def objectives(self) -> None:
        for ev in self.evidence.values():
            for trace_id in ev.langfuse_trace_ids:
                if ev.objective:
                    break
                ev.objective = _objective_of(self.safe_trace(trace_id))
            if not ev.objective:
                # Session-level scores (the Gradio rating) carry no trace id, and
                # resumed segments carry no objective: look across the session.
                ev.objective = self.session_objective(ev.thread_id)

    def session_objective(self, session_id: str) -> str:
        try:
            traces = self.api.trace.list(session_id=session_id, limit=_PAGE_SIZE).data
        except Exception as exc:
            self.warnings.append(
                f"Langfuse session {session_id} lookup skipped ({type(exc).__name__})."
            )
            return ""
        return next((o for o in map(_objective_of, traces) if o), "")


def _objective_of(trace: Any) -> str:
    metadata = getattr(trace, "metadata", None) or {}
    return str(metadata.get("objective") or "") if isinstance(metadata, dict) else ""


def collect_from_langfuse(
    since: datetime, *, api: Any | None = None
) -> tuple[dict[str, RunEvidence], list[str]]:
    try:
        if api is None:
            from market_research_team.observability import langfuse_api

            api = langfuse_api()
        collector = _Collector(api)
        collector.scores(since)
        collector.errors(since)
        collector.objectives()
        return collector.evidence, collector.warnings
    except Exception as exc:
        return {}, [f"Langfuse unavailable ({type(exc).__name__}: {exc}); audit log only."]
