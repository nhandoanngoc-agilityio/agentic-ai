"""Append-only audit log: one JSON line per run-level event.

Written from the supervisor when a run ends (covers the CLI, `langgraph dev`
and the frontend alike) and from the CLI for each human approval decision.
Deliberately minimal -- objective, route trace, guardrail events, outcome --
and never raises: an audit failure must not turn into a run failure.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_research_team.config import settings
from market_research_team.security.patterns import SECRET_PATTERNS

logger = logging.getLogger(__name__)


def _scrub(value: Any) -> Any:
    """Make sure a credential pasted into an objective never lands in the log."""

    if isinstance(value, str):
        for _, regex in SECRET_PATTERNS:
            value = regex.sub("[secret removed]", value)
        return value
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    return value


def _agent_version() -> str:
    """Imported lazily: `versioning` reads the agent modules, which import this one."""

    try:
        from market_research_team.versioning import agent_version

        return agent_version()
    except Exception:
        return "unknown"


def record(event: str, thread_id: str | None, *, path: Path | None = None, **fields: Any) -> None:
    """Append one JSON line describing `event` to the audit log."""

    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "thread_id": thread_id,
        "agent_version": _agent_version(),
        **_scrub(fields),
    }
    target = path or settings.audit_log_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.exception("Audit write failed for event %r", event)


def record_input_rejection(thread_id: str | None, objective: str, error: Exception) -> None:
    """Record an objective rejected before the graph ran (Gradio and the CLI
    validate first), in the same shape `input_guard_node` produces, so the
    failure harvester sees the blocked attempt."""

    record(
        "run_finished",
        thread_id,
        objective=objective,
        route_trace=[],
        error=f"Input rejected: {error}",
        guardrail_events=[{"layer": "input", "rule": "validate_objective", "detail": str(error)}],
        report_path=None,
    )


def satisfaction_rate(path: Path | None = None) -> tuple[float, int]:
    """Fraction of recorded `user_satisfaction` events rated "up", and the
    total count. Returns `(0.0, 0)` when there are no ratings yet (the
    audit log doesn't exist, or has no such events).
    """

    target = path or settings.audit_log_path
    if not target.exists():
        return 0.0, 0

    ratings = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "user_satisfaction":
            ratings.append(entry.get("rating"))

    total = len(ratings)
    if total == 0:
        return 0.0, 0
    up_count = sum(1 for rating in ratings if rating == "up")
    return up_count / total, total


def latency_summary(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Mean duration (ms) and count for each `node_latency` node, plus the
    overall `run_latency`. Reads the same local, offline audit log as
    `satisfaction_rate` -- a signal that exists regardless of whether
    Langfuse tracing is configured. Returns `{}` when there's nothing
    recorded yet.
    """

    target = path or settings.audit_log_path
    if not target.exists():
        return {}

    node_durations: dict[str, list[float]] = {}
    run_durations: list[float] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration_ms = entry.get("duration_ms")
        if not isinstance(duration_ms, (int, float)):
            continue

        if entry.get("event") == "node_latency" and entry.get("node"):
            node_durations.setdefault(entry["node"], []).append(duration_ms)
        elif entry.get("event") == "run_latency":
            run_durations.append(duration_ms)

    summary = {
        node: {"mean_ms": sum(durations) / len(durations), "count": len(durations)}
        for node, durations in node_durations.items()
    }
    if run_durations:
        summary["run"] = {
            "mean_ms": sum(run_durations) / len(run_durations),
            "count": len(run_durations),
        }
    return summary


def fallback_counts(path: Path | None = None) -> dict[str, int]:
    """Count of `fallback_triggered` events per component (e.g.
    "query_rewriter", "supervisor_router", "reporting_draft"). These
    fallbacks degrade a run gracefully rather than failing it, so they
    never surface anywhere else -- this is the only visibility into how
    often they actually fire. Returns `{}` when there's nothing recorded.
    """

    target = path or settings.audit_log_path
    if not target.exists():
        return {}

    counts: dict[str, int] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "fallback_triggered" and entry.get("component"):
            counts[entry["component"]] = counts.get(entry["component"], 0) + 1
    return counts


def token_usage_summary(path: Path | None = None) -> dict[str, dict[str, int]]:
    """Total input/output/total tokens per `llm_usage` component, plus an
    "overall" entry summed across all components. No dollar-cost estimate
    -- per-token pricing varies by provider/model and changes often, so a
    hardcoded table would go stale and mislead; token counts are the
    stable, reliable signal. Returns `{}` when there's nothing recorded.
    """

    target = path or settings.audit_log_path
    if not target.exists():
        return {}

    totals: dict[str, dict[str, int]] = {}
    overall = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "llm_usage" or not entry.get("component"):
            continue

        component_totals = totals.setdefault(
            entry["component"], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = entry.get(key, 0)
            if isinstance(value, int):
                component_totals[key] += value
                overall[key] += value

    if totals:
        totals["overall"] = overall
    return totals


def tool_call_summary(path: Path | None = None) -> dict[str, dict[str, float | int]]:
    """Per-tool mean duration (ms), call count, and outcome breakdown
    (ok/error/unknown_tool counts) from `tool_call` events. Returns `{}`
    when there's nothing recorded yet.
    """

    target = path or settings.audit_log_path
    if not target.exists():
        return {}

    per_tool: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "tool_call" or not entry.get("tool"):
            continue

        duration_ms = entry.get("duration_ms")
        outcome = entry.get("outcome", "unknown")
        stats = per_tool.setdefault(
            entry["tool"],
            {"durations": [], "ok": 0, "error": 0, "unknown_tool": 0},
        )
        if isinstance(duration_ms, (int, float)):
            stats["durations"].append(duration_ms)
        if outcome in ("ok", "error", "unknown_tool"):
            stats[outcome] += 1

    summary: dict[str, dict[str, float | int]] = {}
    for tool, stats in per_tool.items():
        durations = stats["durations"]
        summary[tool] = {
            "mean_ms": sum(durations) / len(durations) if durations else 0.0,
            "count": stats["ok"] + stats["error"] + stats["unknown_tool"],
            "ok": stats["ok"],
            "error": stats["error"],
            "unknown_tool": stats["unknown_tool"],
        }
    return summary


def current_thread_id() -> str | None:
    """Thread id of the running graph invocation, if any."""

    try:
        from langgraph.config import get_config

        return get_config().get("configurable", {}).get("thread_id")
    except Exception:
        return None
