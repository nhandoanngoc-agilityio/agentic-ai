"""Harvest: audit evidence -> triggers -> candidate queue."""

import json
from datetime import UTC, datetime
from pathlib import Path

from market_research_team.feedback.candidates import CandidateQueue
from market_research_team.feedback.harvest import harvest
from market_research_team.feedback.signals import RunEvidence

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)


def _audit(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def _down(thread: str, ts: str, objective: str = "Compare Acme pricing") -> dict:
    return {
        "ts": ts,
        "event": "user_satisfaction",
        "thread_id": thread,
        "objective": objective,
        "rating": "down",
    }


def test_harvest_creates_candidates_only_for_failing_runs(tmp_path: Path):
    ok = {
        "ts": "2026-09-29T10:00:00+00:00",
        "event": "user_satisfaction",
        "thread_id": "t2",
        "objective": "fine",
        "rating": "up",
    }
    audit = _audit(tmp_path / "a.jsonl", [_down("t1", "2026-09-29T10:00:00+00:00"), ok])
    queue = CandidateQueue(tmp_path / "q")

    report = harvest(queue, audit_path=audit, now=NOW)

    assert len(report.new) == 1 and report.runs_scanned == 2
    assert queue.last_harvest() == NOW.isoformat()


def test_harvest_twice_does_not_duplicate_evidence(tmp_path: Path):
    audit = _audit(
        tmp_path / "a.jsonl",
        [
            _down("t1", "2026-09-29T10:00:00+00:00"),
            _down("t2", "2026-09-29T11:00:00+00:00", objective="compare  ACME pricing"),
        ],
    )
    queue = CandidateQueue(tmp_path / "q")
    since = datetime(2026, 9, 1, tzinfo=UTC)

    harvest(queue, audit_path=audit, since=since, now=NOW)
    second = harvest(queue, audit_path=audit, since=since, now=NOW)

    (candidate,), _ = queue.list_pending()
    assert candidate.occurrences == 2
    assert len(candidate.evidence) == 2
    assert second.new == []


def test_default_since_is_last_harvest(tmp_path: Path):
    audit = _audit(tmp_path / "a.jsonl", [_down("t1", "2026-09-20T10:00:00+00:00")])
    queue = CandidateQueue(tmp_path / "q")
    queue.set_last_harvest("2026-09-25T00:00:00+00:00")

    assert harvest(queue, audit_path=audit, now=NOW).runs_scanned == 0


def test_bounded_harvest_does_not_move_the_marker(tmp_path: Path):
    audit = _audit(tmp_path / "a.jsonl", [_down("t1", "2026-06-01T10:00:00+00:00")])
    queue = CandidateQueue(tmp_path / "q")

    harvest(
        queue,
        audit_path=audit,
        since=datetime(2026, 1, 1, tzinfo=UTC),
        until=datetime(2026, 7, 1, tzinfo=UTC),
        now=NOW,
    )

    assert queue.last_harvest() is None
    assert len(queue.list_pending()[0]) == 1


def test_langfuse_evidence_is_merged_and_warnings_reported(tmp_path: Path):
    audit = _audit(tmp_path / "a.jsonl", [])
    queue = CandidateQueue(tmp_path / "q")

    def fake_langfuse(since):
        ev = RunEvidence("t9", objective="From Langfuse", ratings=["down"], sources=["langfuse"])
        return {"t9": ev}, ["partial: cap hit"]

    report = harvest(queue, audit_path=audit, langfuse=fake_langfuse, now=NOW)

    assert len(report.new) == 1
    assert report.warnings == ["partial: cap hit"]


def test_rejection_before_the_run_finishes_is_harvested(tmp_path: Path):
    # The objective comes from the run_latency of the interrupted first segment;
    # run_finished only arrives after the resume.
    audit = _audit(
        tmp_path / "a.jsonl",
        [
            {
                "ts": "2026-09-29T09:59:00+00:00",
                "event": "run_latency",
                "thread_id": "t1",
                "objective": "Compare Acme pricing",
                "duration_ms": 10,
                "outcome": "interrupted",
            },
            {
                "ts": "2026-09-29T10:00:00+00:00",
                "event": "human_decision",
                "thread_id": "t1",
                "decision": {"approved": False, "feedback": "wrong"},
            },
        ],
    )
    queue = CandidateQueue(tmp_path / "q")

    report = harvest(queue, audit_path=audit, since=datetime(2026, 9, 1, tzinfo=UTC), now=NOW)

    assert len(report.new) == 1


def test_failing_run_without_an_objective_is_reported(tmp_path: Path):
    audit = _audit(
        tmp_path / "a.jsonl",
        [
            {
                "ts": "2026-09-29T10:00:00+00:00",
                "event": "human_decision",
                "thread_id": "t1",
                "decision": {"approved": False},
            }
        ],
    )
    queue = CandidateQueue(tmp_path / "q")

    report = harvest(queue, audit_path=audit, since=datetime(2026, 9, 1, tzinfo=UTC), now=NOW)

    assert report.new == []
    assert any("no objective" in w for w in report.warnings)


def test_events_stamped_just_before_the_marker_are_reread(tmp_path: Path):
    # Audit timestamps are whole seconds; the marker has microseconds.
    audit = _audit(tmp_path / "a.jsonl", [_down("t1", "2026-09-29T10:00:00+00:00")])
    queue = CandidateQueue(tmp_path / "q")
    queue.set_last_harvest("2026-09-29T10:00:00.500000+00:00")

    assert len(harvest(queue, audit_path=audit, now=NOW).new) == 1


def test_langfuse_window_overlaps_the_marker_by_an_hour(tmp_path: Path):
    queue = CandidateQueue(tmp_path / "q")
    queue.set_last_harvest("2026-09-29T10:00:00+00:00")
    seen: list = []

    def fake_langfuse(since):
        seen.append(since)
        return {}, []

    harvest(queue, audit_path=tmp_path / "none.jsonl", langfuse=fake_langfuse, now=NOW)

    assert seen == [datetime(2026, 9, 29, 9, 0, tzinfo=UTC)]


def test_corrupt_pending_file_is_skipped_with_a_warning(tmp_path: Path):
    from market_research_team.feedback.candidates import candidate_id

    audit = _audit(tmp_path / "a.jsonl", [_down("t1", "2026-09-29T10:00:00+00:00")])
    queue = CandidateQueue(tmp_path / "q")
    pending = tmp_path / "q" / "pending"
    pending.mkdir(parents=True)
    (pending / f"{candidate_id('Compare Acme pricing')}.json").write_text("{nope")

    report = harvest(queue, audit_path=audit, since=datetime(2026, 9, 1, tzinfo=UTC), now=NOW)

    assert any("invalid candidate file" in w for w in report.warnings)
