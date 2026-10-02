"""Harvest production failures into the candidate queue (audit log + Langfuse)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from market_research_team.feedback.candidates import CandidateQueue, candidate_id
from market_research_team.feedback.signals import (
    RunEvidence,
    collect_from_audit,
    merge_evidence,
    parse_ts,
)
from market_research_team.feedback.triggers import detect_triggers

FIRST_HARVEST_WINDOW = timedelta(days=90)
# Re-read a little before the last harvest: audit timestamps are whole seconds
# and lines can land while a harvest runs; Langfuse ingests asynchronously.
# Re-reading is safe because the queue deduplicates by thread.
AUDIT_OVERLAP = timedelta(minutes=10)
LANGFUSE_OVERLAP = timedelta(hours=1)
LangfuseCollector = Callable[[datetime], tuple[dict[str, RunEvidence], list[str]]]


@dataclass
class HarvestReport:
    new: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runs_scanned: int = 0


def harvest(
    queue: CandidateQueue,
    *,
    audit_path: Path,
    since: datetime | None = None,
    until: datetime | None = None,
    langfuse: LangfuseCollector | None = None,
    now: datetime | None = None,
) -> HarvestReport:
    """Collect evidence, keep runs with a failure trigger, queue them.

    `since=None` continues from the last harvest (or the last 90 days). A
    bounded harvest (`until` set, as pruning does) leaves the marker alone.
    """

    now = now or datetime.now(UTC)
    langfuse_since = since
    if since is None:
        marker = parse_ts(queue.last_harvest() or "")
        if marker is None:
            since = langfuse_since = now - FIRST_HARVEST_WINDOW
        else:
            since, langfuse_since = marker - AUDIT_OVERLAP, marker - LANGFUSE_OVERLAP
    report = HarvestReport()
    groups = [collect_from_audit(audit_path, since=since, until=until)]
    if langfuse is not None:
        evidence, warnings = langfuse(langfuse_since or since)
        groups.append(evidence)
        report.warnings += warnings
    runs = merge_evidence(*groups)
    report.runs_scanned = len(runs)
    no_objective = 0
    for ev in runs.values():
        triggers = detect_triggers(ev)
        if not triggers:
            continue
        if not ev.objective:
            no_objective += 1
            continue
        outcome = queue.add(ev, triggers)
        if outcome == "invalid":
            report.warnings.append(
                f"Skipped run {ev.thread_id}: invalid candidate file "
                f"{candidate_id(ev.objective)}.json in pending/ (fix or delete it)."
            )
            continue
        getattr(report, outcome).append(ev.thread_id)
    if no_objective:
        report.warnings.append(
            f"{no_objective} failing run(s) had no objective in the window and were skipped."
        )
    if until is None:
        queue.set_last_harvest(now.isoformat())
    return report
