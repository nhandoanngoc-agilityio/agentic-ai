"""Retention: prune old audit lines and checkpoint threads, never losing a failure."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from market_research_team.feedback import retention
from market_research_team.feedback.candidates import CandidateQueue

CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)


def _audit(path: Path) -> Path:
    lines = [
        json.dumps(
            {
                "ts": "2026-06-01T00:00:00+00:00",
                "event": "user_satisfaction",
                "thread_id": "old",
                "objective": "Old failing objective",
                "rating": "down",
            }
        ),
        json.dumps({"ts": "2026-09-01T00:00:00+00:00", "event": "run_latency", "thread_id": "new"}),
        "not json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_prune_audit_log_dry_run_changes_nothing(tmp_path: Path):
    log = _audit(tmp_path / "a.jsonl")
    before = log.read_text()

    assert retention.prune_audit_log(log, CUTOFF, apply=False) == (2, 1)
    assert log.read_text() == before


def test_prune_audit_log_keeps_recent_and_unparseable(tmp_path: Path):
    log = _audit(tmp_path / "a.jsonl")

    assert retention.prune_audit_log(log, CUTOFF, apply=True) == (2, 1)
    text = log.read_text()
    assert "Old failing objective" not in text and "run_latency" in text and "not json" in text


def test_audit_rewrite_is_atomic_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log = _audit(tmp_path / "a.jsonl")
    before = log.read_text()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(retention.os, "replace", boom)
    with pytest.raises(OSError):
        retention.prune_audit_log(log, CUTOFF, apply=True)

    assert log.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.jsonl"]  # temp file cleaned up


def _saver_with_threads() -> InMemorySaver:
    class _S(TypedDict):
        x: int

    def ask(_state):
        interrupt({"approve?": True})
        return {"x": 1}

    saver = InMemorySaver()
    plain = StateGraph(_S)
    plain.add_node("n", lambda s: {"x": 1})
    plain.add_edge(START, "n")
    plain.add_edge("n", END)
    paused = StateGraph(_S)
    paused.add_node("n", ask)
    paused.add_edge(START, "n")
    paused.add_edge("n", END)
    plain.compile(checkpointer=saver).invoke({"x": 0}, {"configurable": {"thread_id": "done"}})
    paused.compile(checkpointer=saver).invoke({"x": 0}, {"configurable": {"thread_id": "paused"}})
    return saver


def test_prune_checkpoints_deletes_only_old_threads_and_counts_paused():
    saver = _saver_with_threads()
    future = datetime(2100, 1, 1, tzinfo=UTC)

    assert retention.prune_checkpoints(saver, CUTOFF, apply=True) == (0, 0)
    assert retention.prune_checkpoints(saver, future, apply=False) == (2, 1)
    assert len(list(saver.list(None))) > 0
    assert retention.prune_checkpoints(saver, future, apply=True) == (2, 1)
    assert list(saver.list(None)) == []


def test_prune_harvests_before_deleting(tmp_path: Path):
    log = _audit(tmp_path / "a.jsonl")
    queue = CandidateQueue(tmp_path / "q")

    report = retention.prune(CUTOFF, apply=True, audit_path=log, saver=InMemorySaver(), queue=queue)

    assert report.candidates_harvested == 1
    (candidate,), _ = queue.list_pending()
    assert candidate.objective == "Old failing objective"
    assert "Old failing objective" not in log.read_text()
    assert queue.last_harvest() is None  # bounded harvest leaves the marker alone


def test_dry_run_prune_writes_nothing(tmp_path: Path):
    log = _audit(tmp_path / "a.jsonl")
    queue = CandidateQueue(tmp_path / "q")

    report = retention.prune(
        CUTOFF, apply=False, audit_path=log, saver=InMemorySaver(), queue=queue
    )

    assert report.applied is False and report.audit_removed == 1
    assert queue.list_pending() == ([], [])


NOW = datetime(2026, 9, 30, tzinfo=UTC)


@pytest.fixture
def marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from market_research_team.config import settings

    path = tmp_path / ".last_prune"
    monkeypatch.setattr(settings, "prune_marker_path", path)
    monkeypatch.setattr(settings, "auto_prune_enabled", True)
    monkeypatch.setattr(settings, "audit_retention_days", 90)
    return path


def _spy():
    calls: list = []

    def fake_prune(cutoff, *, apply):
        calls.append((cutoff, apply))
        return retention.PruneReport(applied=apply)

    return calls, fake_prune


def test_auto_prune_runs_once_per_day(marker: Path):
    calls, fake = _spy()

    assert retention.maybe_auto_prune(now=NOW, allow_in_tests=True, prune_fn=fake) is not None
    assert calls == [(datetime(2026, 7, 2, tzinfo=UTC), True)]
    assert marker.exists()
    assert retention.maybe_auto_prune(now=NOW, allow_in_tests=True, prune_fn=fake) is None
    assert len(calls) == 1


def test_auto_prune_respects_disable_flag_and_pytest_guard(marker: Path, monkeypatch):
    from market_research_team.config import settings

    calls, fake = _spy()
    assert retention.maybe_auto_prune(now=NOW, prune_fn=fake) is None  # under pytest
    monkeypatch.setattr(settings, "auto_prune_enabled", False)
    assert retention.maybe_auto_prune(now=NOW, allow_in_tests=True, prune_fn=fake) is None
    assert calls == []


def test_auto_prune_swallows_errors(marker: Path):
    def boom(cutoff, *, apply):
        raise RuntimeError("db locked")

    assert retention.maybe_auto_prune(now=NOW, allow_in_tests=True, prune_fn=boom) is None
    assert not marker.exists()
