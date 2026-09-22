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


def record(event: str, thread_id: str | None, *, path: Path | None = None, **fields: Any) -> None:
    """Append one JSON line describing `event` to the audit log."""

    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "thread_id": thread_id,
        **_scrub(fields),
    }
    target = path or settings.audit_log_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.exception("Audit write failed for event %r", event)


def current_thread_id() -> str | None:
    """Thread id of the running graph invocation, if any."""

    try:
        from langgraph.config import get_config

        return get_config().get("configurable", {}).get("thread_id")
    except Exception:
        return None
