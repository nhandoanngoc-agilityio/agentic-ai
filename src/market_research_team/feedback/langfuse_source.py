"""Pull production failure signals from Langfuse into RunEvidence.

Two signals, merged with the audit log by thread_id (== Langfuse session_id):
`user_feedback` scores of 0 (thumbs down, see
observability.record_feedback_score) and ERROR-level observations. Any API
failure returns no evidence plus a warning, so a harvest always falls back to
the audit log.

Only current read APIs: scores come from `GET /api/public/v3/scores` and trace
lookups from `GET /api/public/v2/observations`. The legacy `v2/scores` and
`traces` endpoints return 410 for organizations created on or after
2026-09-16. Trace metadata (the run's `objective`) is propagated onto every
observation, so observations are enough to recover it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from market_research_team.feedback.signals import EVAL_THREAD_PREFIX, RunEvidence

MAX_ITEMS = 1000
_PAGE_SIZE = 100


# Observation field groups for lookups: `basic` carries sessionId, `metadata` the objective.
_LOOKUP_FIELDS = "core,basic,metadata"


def _iso(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


class _Collector:
    def __init__(self, api: Any) -> None:
        self.api = api
        self.evidence: dict[str, RunEvidence] = {}
        self.traces: dict[str, Any] = {}
        self.warnings: list[str] = []

    def _observations(self, **filters: Any) -> list[Any]:
        return self.api.observations.get_many(
            fields=_LOOKUP_FIELDS, expand_metadata="objective", limit=_PAGE_SIZE, **filters
        ).data

    def trace(self, trace_id: str) -> Any:
        """The trace's session id and objective, rebuilt from its observations."""

        if trace_id not in self.traces:
            rows = self._observations(trace_id=trace_id)
            session_id = next((r.session_id for r in rows if getattr(r, "session_id", None)), None)
            objective = next((o for o in map(_objective_of, rows) if o), "")
            self.traces[trace_id] = _Trace(session_id, objective)
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
        cursor, seen = None, 0
        while True:
            # Thumbs down is a NUMERIC 0.0 (`record_feedback_score`); `subject`
            # says what the score is attached to (session, trace or observation).
            response = self.api.scores_v3.get_many_v3(
                name="user_feedback",
                data_type="NUMERIC",
                value_max=0.0,
                from_timestamp=since,
                fields="subject",
                limit=_PAGE_SIZE,
                cursor=cursor,
            )
            for score in response.data:
                if seen >= MAX_ITEMS:
                    self.warnings.append(f"Langfuse scores: stopped at the {MAX_ITEMS}-item cap.")
                    return
                seen += 1
                ev = self.run_for(*_session_and_trace(getattr(score, "subject", None)))
                if ev is not None:
                    ev.ratings.append("down")
                    ev.seen(_iso(getattr(score, "timestamp", None)))
            cursor = getattr(response.meta, "cursor", None)
            if not cursor:
                return

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
            rows = self._observations(session_id=session_id)
        except Exception as exc:
            self.warnings.append(
                f"Langfuse session {session_id} lookup skipped ({type(exc).__name__})."
            )
            return ""
        return next((o for o in map(_objective_of, rows) if o), "")


class _Trace:
    """What the harvester needs from a trace: its session and objective."""

    def __init__(self, session_id: str | None, objective: str) -> None:
        self.session_id = session_id
        self.metadata = {"objective": objective} if objective else {}


def _session_and_trace(subject: Any) -> tuple[str | None, str | None]:
    """(session_id, trace_id) from a v3 score's `subject`."""

    kind = getattr(subject, "kind", None)
    if kind == "session":
        return subject.id, None
    if kind == "trace":
        return None, subject.id
    if kind == "observation":
        return None, getattr(subject, "trace_id", None)
    return None, None  # experiment scores, or no subject: not a production run


def _objective_of(row: Any) -> str:
    metadata = getattr(row, "metadata", None) or {}
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
