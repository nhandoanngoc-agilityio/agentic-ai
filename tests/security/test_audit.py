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


def test_satisfaction_rate_is_zero_when_log_missing(tmp_path: Path) -> None:
    rate, count = audit.satisfaction_rate(tmp_path / "missing.jsonl")
    assert rate == 0.0
    assert count == 0


def test_satisfaction_rate_computes_up_fraction(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record("user_satisfaction", "t1", path=log, objective="o1", rating="up")
    audit.record("user_satisfaction", "t2", path=log, objective="o2", rating="down")
    audit.record("user_satisfaction", "t3", path=log, objective="o3", rating="up")
    audit.record("run_finished", "t4", path=log, objective="o4")

    rate, count = audit.satisfaction_rate(log)

    assert count == 3
    assert rate == 2 / 3


def test_satisfaction_rate_ignores_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record("user_satisfaction", "t1", path=log, objective="o1", rating="up")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    rate, count = audit.satisfaction_rate(log)

    assert count == 1
    assert rate == 1.0


def test_latency_summary_is_empty_when_log_missing(tmp_path: Path) -> None:
    assert audit.latency_summary(tmp_path / "missing.jsonl") == {}


def test_latency_summary_computes_mean_per_node_and_run(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record("node_latency", "t1", path=log, node="research", duration_ms=100.0, outcome="ok")
    audit.record("node_latency", "t1", path=log, node="research", duration_ms=200.0, outcome="ok")
    audit.record("node_latency", "t1", path=log, node="analytics", duration_ms=50.0, outcome="ok")
    audit.record(
        "run_latency", "t1", path=log, objective="o", duration_ms=500.0, outcome="finished"
    )

    summary = audit.latency_summary(log)

    assert summary["research"] == {"mean_ms": 150.0, "count": 2}
    assert summary["analytics"] == {"mean_ms": 50.0, "count": 1}
    assert summary["run"] == {"mean_ms": 500.0, "count": 1}


def test_latency_summary_ignores_malformed_and_unrelated_lines(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record("node_latency", "t1", path=log, node="research", duration_ms=100.0, outcome="ok")
    audit.record("run_finished", "t1", path=log, objective="o")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    summary = audit.latency_summary(log)

    assert summary == {"research": {"mean_ms": 100.0, "count": 1}}


def test_fallback_counts_is_empty_when_log_missing(tmp_path: Path) -> None:
    assert audit.fallback_counts(tmp_path / "missing.jsonl") == {}


def test_fallback_counts_groups_by_component(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record(
        "fallback_triggered", "t1", path=log, component="query_rewriter", reason="exception: boom"
    )
    audit.record(
        "fallback_triggered", "t1", path=log, component="query_rewriter", reason="empty_result"
    )
    audit.record(
        "fallback_triggered", "t1", path=log, component="reporting_draft", reason="empty_result"
    )
    audit.record("run_finished", "t1", path=log, objective="o")

    counts = audit.fallback_counts(log)

    assert counts == {"query_rewriter": 2, "reporting_draft": 1}


def test_token_usage_summary_is_empty_when_log_missing(tmp_path: Path) -> None:
    assert audit.token_usage_summary(tmp_path / "missing.jsonl") == {}


def test_token_usage_summary_totals_per_component_and_overall(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record(
        "llm_usage",
        "t1",
        path=log,
        component="query_rewriter",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    audit.record(
        "llm_usage",
        "t1",
        path=log,
        component="query_rewriter",
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
    )
    audit.record(
        "llm_usage",
        "t1",
        path=log,
        component="reporting_draft",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )
    audit.record("run_finished", "t1", path=log, objective="o")

    summary = audit.token_usage_summary(log)

    assert summary["query_rewriter"] == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }
    assert summary["reporting_draft"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert summary["overall"] == {
        "input_tokens": 130,
        "output_tokens": 65,
        "total_tokens": 195,
    }


def test_token_usage_summary_ignores_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record(
        "llm_usage",
        "t1",
        path=log,
        component="query_rewriter",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    summary = audit.token_usage_summary(log)

    assert summary["query_rewriter"]["total_tokens"] == 15


def test_tool_call_summary_is_empty_when_log_missing(tmp_path: Path) -> None:
    assert audit.tool_call_summary(tmp_path / "missing.jsonl") == {}


def test_tool_call_summary_aggregates_duration_and_outcomes(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit.record("tool_call", "t1", path=log, tool="mean", duration_ms=10.0, outcome="ok")
    audit.record("tool_call", "t1", path=log, tool="mean", duration_ms=30.0, outcome="ok")
    audit.record("tool_call", "t1", path=log, tool="mean", duration_ms=5.0, outcome="error")
    audit.record(
        "tool_call", "t1", path=log, tool="made_up", duration_ms=0.0, outcome="unknown_tool"
    )
    audit.record("run_finished", "t1", path=log, objective="o")

    summary = audit.tool_call_summary(log)

    assert summary["mean"] == {"mean_ms": 15.0, "count": 3, "ok": 2, "error": 1, "unknown_tool": 0}
    assert summary["made_up"] == {
        "mean_ms": 0.0,
        "count": 1,
        "ok": 0,
        "error": 0,
        "unknown_tool": 1,
    }


def test_record_stamps_the_agent_version(tmp_path: Path) -> None:
    from market_research_team.versioning import agent_version

    log = tmp_path / "audit.jsonl"
    audit.record("run_finished", "thread-1", path=log)

    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry["agent_version"] == agent_version()
