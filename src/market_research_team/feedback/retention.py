"""Retention: drop audit lines, checkpoint threads and remembered reviewer
notes older than the cutoff.

Harvest first: failures in the window about to be deleted become regression
candidates before their evidence goes (candidates keep their own copy).
Langfuse data is not touched; its retention is configured in Langfuse.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from market_research_team.config import settings
from market_research_team.feedback.candidates import CandidateQueue
from market_research_team.feedback.harvest import harvest
from market_research_team.feedback.signals import parse_ts
from market_research_team.memory import prune_memory


@dataclass
class PruneReport:
    applied: bool = False
    audit_kept: int = 0
    audit_removed: int = 0
    threads_deleted: int = 0
    paused_threads_deleted: int = 0
    candidates_harvested: int = 0
    memory_notes_removed: int = 0


def _is_old(line: str, cutoff: datetime) -> bool:
    try:
        ts = parse_ts(str(json.loads(line).get("ts", "")))
    except (json.JSONDecodeError, AttributeError):
        return False
    return ts is not None and ts < cutoff


def prune_audit_log(path: Path, cutoff: datetime, *, apply: bool) -> tuple[int, int]:
    """Returns (kept, removed). Unparseable lines are kept. The rewrite is
    atomic: a crash leaves the original file untouched."""

    if not path.exists():
        return 0, 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines if not _is_old(line, cutoff)]
    removed = len(lines) - len(kept)
    if apply and removed:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in kept))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return len(kept), removed


def prune_checkpoints(saver: Any, cutoff: datetime, *, apply: bool) -> tuple[int, int]:
    """Delete threads whose latest checkpoint is older than the cutoff.
    Returns (deleted, of which paused on an approval interrupt)."""

    latest: dict[str, tuple[datetime, Any]] = {}
    for tup in saver.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        ts = parse_ts(tup.checkpoint.get("ts", ""))
        if ts is None:
            continue
        if thread_id not in latest or ts > latest[thread_id][0]:
            latest[thread_id] = (ts, tup)
    deleted = paused = 0
    for thread_id, (ts, tup) in latest.items():
        if ts >= cutoff:
            continue
        deleted += 1
        if any(write[1] == "__interrupt__" for write in (tup.pending_writes or [])):
            paused += 1
        if apply:
            saver.delete_thread(thread_id)
    return deleted, paused


def prune(
    cutoff: datetime,
    *,
    apply: bool,
    audit_path: Path | None = None,
    saver: Any | None = None,
    queue: CandidateQueue | None = None,
    store: Any | None = None,
) -> PruneReport:
    audit_path = audit_path or settings.audit_log_path
    queue = queue or CandidateQueue(settings.candidates_dir)
    if saver is None:
        from market_research_team.checkpointing.store import get_checkpointer

        saver = get_checkpointer()
    report = PruneReport(applied=apply)
    if apply:
        harvested = harvest(
            queue,
            audit_path=audit_path,
            since=datetime.min.replace(tzinfo=cutoff.tzinfo),
            until=cutoff,
        )
        report.candidates_harvested = len(harvested.new) + len(harvested.merged)
    report.audit_kept, report.audit_removed = prune_audit_log(audit_path, cutoff, apply=apply)
    report.threads_deleted, report.paused_threads_deleted = prune_checkpoints(
        saver, cutoff, apply=apply
    )
    if store is None:
        from market_research_team.checkpointing.store import get_memory_store

        store = get_memory_store()
    report.memory_notes_removed = prune_memory(store, cutoff, apply=apply)
    return report


logger = logging.getLogger(__name__)
_AUTO_PRUNE_INTERVAL = timedelta(hours=24)


def maybe_auto_prune(
    *, now: datetime | None = None, allow_in_tests: bool = False, prune_fn: Any = None
) -> PruneReport | None:
    """Prune at app start, at most once per 24 h. Never raises: a failure is
    logged and the app starts normally. Skipped under pytest."""

    if not settings.auto_prune_enabled:
        return None
    if os.environ.get("PYTEST_CURRENT_TEST") and not allow_in_tests:
        return None
    now = now or datetime.now(UTC)
    marker = settings.prune_marker_path
    try:
        if marker.exists():
            last = parse_ts(marker.read_text(encoding="utf-8").strip())
            if last is not None and now - last < _AUTO_PRUNE_INTERVAL:
                return None
        cutoff = now - timedelta(days=settings.audit_retention_days)
        report = (prune_fn or prune)(cutoff, apply=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(now.isoformat() + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning("Auto-prune skipped: %s", exc)
        return None
    logger.info(
        "Auto-prune: %d audit lines removed, %d threads deleted (%d paused), "
        "%d reviewer notes removed, %d candidates harvested",
        report.audit_removed,
        report.threads_deleted,
        report.paused_threads_deleted,
        report.memory_notes_removed,
        report.candidates_harvested,
    )
    return report
