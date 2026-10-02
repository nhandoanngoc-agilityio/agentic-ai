"""Grouping production audit events into per-run evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from market_research_team.feedback.signals import (
    RunEvidence,
    collect_from_audit,
    merge_evidence,
    read_audit_events,
)


def _log(path: Path, events: list[dict]) -> Path:
    lines = [json.dumps(e) for e in events] + ["not json", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_read_audit_events_skips_malformed_lines(tmp_path: Path):
    log = _log(tmp_path / "a.jsonl", [{"event": "x"}])

    assert read_audit_events(log) == [{"event": "x"}]
    assert read_audit_events(tmp_path / "missing.jsonl") == []


def test_collect_groups_events_by_thread(tmp_path: Path):
    log = _log(
        tmp_path / "a.jsonl",
        [
            {
                "ts": "2026-09-20T10:00:00+00:00",
                "event": "run_finished",
                "thread_id": "t1",
                "agent_version": "0.1.0+aaa",
                "objective": "Compare Acme",
                "route_trace": ["research", "reporting"],
                "guardrail_events": [
                    {"layer": "retrieval", "rule": "injection_in_chunk", "detail": "d"}
                ],
                "error": None,
                "report_path": "reports/a.md",
            },
            {
                "ts": "2026-09-20T10:01:00+00:00",
                "event": "human_decision",
                "thread_id": "t1",
                "decision": {"approved": False, "feedback": "wrong"},
            },
            {
                "ts": "2026-09-20T10:02:00+00:00",
                "event": "user_satisfaction",
                "thread_id": "t1",
                "objective": "Compare Acme",
                "rating": "down",
            },
            {
                "ts": "2026-09-20T10:00:30+00:00",
                "event": "fallback_triggered",
                "thread_id": "t1",
                "component": "reporting_draft",
                "reason": "x",
            },
            {
                "ts": "2026-09-20T11:00:00+00:00",
                "event": "run_finished",
                "thread_id": "eval-x-r0",
                "objective": "ignored",
            },
            {"ts": "2026-09-20T11:00:00+00:00", "event": "llm_usage", "thread_id": None},
        ],
    )

    evidence = collect_from_audit(log)

    assert set(evidence) == {"t1"}
    ev = evidence["t1"]
    assert ev.objective == "Compare Acme"
    assert ev.agent_version == "0.1.0+aaa"
    assert ev.route_trace == ["research", "reporting"]
    assert ev.guardrail_events[0]["rule"] == "injection_in_chunk"
    assert ev.decisions == [{"approved": False, "feedback": "wrong"}]
    assert ev.ratings == ["down"]
    assert ev.fallbacks == ["reporting_draft"]
    assert ev.first_seen == "2026-09-20T10:00:00+00:00"
    assert ev.last_seen == "2026-09-20T10:02:00+00:00"
    assert ev.sources == ["audit"]


def test_collect_filters_by_time_window(tmp_path: Path):
    log = _log(
        tmp_path / "a.jsonl",
        [
            {
                "ts": "2026-09-01T00:00:00+00:00",
                "event": "user_satisfaction",
                "thread_id": "old",
                "objective": "o",
                "rating": "down",
            },
            {
                "ts": "2026-09-25T00:00:00+00:00",
                "event": "user_satisfaction",
                "thread_id": "new",
                "objective": "o",
                "rating": "down",
            },
        ],
    )
    since = datetime(2026, 9, 10, tzinfo=UTC)

    assert set(collect_from_audit(log, since=since)) == {"new"}
    assert set(collect_from_audit(log, until=since)) == {"old"}


def test_merge_evidence_combines_sources_without_duplicates():
    a = RunEvidence("t1", objective="o", ratings=["down"], sources=["audit"])
    b = RunEvidence("t1", ratings=["down"], langfuse_trace_ids=["tr1"], sources=["langfuse"])
    c = RunEvidence("t2", objective="p", sources=["langfuse"])

    merged = merge_evidence({"t1": a}, {"t1": b, "t2": c})

    assert set(merged) == {"t1", "t2"}
    assert merged["t1"].objective == "o"
    assert merged["t1"].ratings == ["down"]
    assert merged["t1"].langfuse_trace_ids == ["tr1"]
    assert merged["t1"].sources == ["audit", "langfuse"]
