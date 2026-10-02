"""Group production audit events into one evidence record per run (thread).

Pure: reads a JSONL file, no network, no LLM. Eval runs (thread ids starting
`eval-`) and events without a thread are ignored -- they are not production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_THREAD_PREFIX = "eval-"


@dataclass
class RunEvidence:
    thread_id: str
    objective: str = ""
    agent_version: str = ""
    first_seen: str = ""
    last_seen: str = ""
    route_trace: list[str] = field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    report_path: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    ratings: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    langfuse_trace_ids: list[str] = field(default_factory=list)
    langfuse_errors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def seen(self, ts: str) -> None:
        if not ts:
            return
        if not self.first_seen or ts < self.first_seen:
            self.first_seen = ts
        if ts > self.last_seen:
            self.last_seen = ts

    def merge(self, other: RunEvidence) -> None:
        """Fold `other` (same thread, another source) into this record."""

        self.objective = self.objective or other.objective
        self.agent_version = self.agent_version or other.agent_version
        self.error = self.error or other.error
        self.report_path = self.report_path or other.report_path
        self.route_trace = self.route_trace or other.route_trace
        self.seen(other.first_seen)
        self.seen(other.last_seen)
        for name in (
            "guardrail_events",
            "decisions",
            "ratings",
            "fallbacks",
            "langfuse_trace_ids",
            "langfuse_errors",
            "sources",
        ):
            mine = getattr(self, name)
            for item in getattr(other, name):
                if item not in mine:
                    mine.append(item)


def parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def read_audit_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _in_window(ts: str, since: datetime | None, until: datetime | None) -> bool:
    if since is None and until is None:
        return True
    parsed = parse_ts(ts)
    if parsed is None:
        return False
    return (since is None or parsed >= since) and (until is None or parsed < until)


def collect_from_audit(
    path: Path, *, since: datetime | None = None, until: datetime | None = None
) -> dict[str, RunEvidence]:
    evidence: dict[str, RunEvidence] = {}
    for event in read_audit_events(path):
        thread_id = event.get("thread_id")
        if not thread_id or str(thread_id).startswith(EVAL_THREAD_PREFIX):
            continue
        ts = str(event.get("ts", ""))
        if not _in_window(ts, since, until):
            continue
        ev = evidence.setdefault(thread_id, RunEvidence(thread_id, sources=["audit"]))
        ev.seen(ts)
        ev.agent_version = event.get("agent_version") or ev.agent_version
        kind = event.get("event")
        if kind == "run_finished":
            ev.objective = event.get("objective") or ev.objective
            ev.route_trace = list(event.get("route_trace") or ev.route_trace)
            ev.guardrail_events += list(event.get("guardrail_events") or [])
            ev.error = event.get("error") or ev.error
            ev.report_path = event.get("report_path") or ev.report_path
        elif kind == "user_satisfaction":
            ev.objective = ev.objective or event.get("objective") or ""
            ev.ratings.append(str(event.get("rating")))
        elif kind == "human_decision":
            ev.decisions.append(dict(event.get("decision") or {}))
        elif kind == "run_latency":
            # Written for every run_graph segment, including the first one that
            # stops at the approval interrupt, so it carries the objective before
            # run_finished exists (or when the run is abandoned).
            ev.objective = ev.objective or event.get("objective") or ""
        elif kind == "fallback_triggered":
            ev.fallbacks.append(str(event.get("component", "unknown")))
    return evidence


def merge_evidence(*groups: dict[str, RunEvidence]) -> dict[str, RunEvidence]:
    merged: dict[str, RunEvidence] = {}
    for group in groups:
        for thread_id, ev in group.items():
            if thread_id in merged:
                merged[thread_id].merge(ev)
            else:
                merged[thread_id] = ev
    return merged
