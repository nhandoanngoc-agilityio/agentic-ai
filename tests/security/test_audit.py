import json
from pathlib import Path

from market_research_team.security import audit


def test_record_appends_one_json_line_per_event(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "audit.jsonl"
    audit.record("run_finished", "thread-1", path=log, objective="Assess Acme", error=None)
    audit.record("human_decision", "thread-1", path=log, decision={"approved": True})

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "run_finished"
    assert first["thread_id"] == "thread-1"
    assert first["objective"] == "Assess Acme"
    assert "ts" in first
    assert second["decision"] == {"approved": True}


def test_record_scrubs_secrets_from_fields(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record(
        "run_finished",
        None,
        path=log,
        objective="Use sk-ant-abcdefghijklmnopqrstuvwxyz0123 to assess Acme",
        guardrail_events=[{"detail": "key sk-proj-abcdefghijklmnopqrstuvwxyz"}],
    )
    entry = json.loads(log.read_text(encoding="utf-8"))
    assert "sk-ant-" not in entry["objective"]
    assert "sk-proj-" not in entry["guardrail_events"][0]["detail"]


def test_record_never_raises_when_path_is_unwritable(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    audit.record("run_finished", "t", path=blocker / "audit.jsonl")


def test_current_thread_id_is_none_outside_a_graph_run() -> None:
    assert audit.current_thread_id() is None
