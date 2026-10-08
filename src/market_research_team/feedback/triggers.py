"""Rules that flag a production run as a failure worth a regression case."""

from __future__ import annotations

from typing import Any

from market_research_team.feedback.signals import RunEvidence
from market_research_team.guardrails import TRANSIENT_MARK

_INPUT_REJECTED_PREFIX = "Input rejected"


def detect_triggers(ev: RunEvidence) -> list[str]:
    triggers: list[str] = []
    if "down" in ev.ratings:
        triggers.append("thumbs_down")
    rejections = [d for d in ev.decisions if d.get("approved") is False and not d.get("discard")]
    if rejections:
        triggers.append("rejected")
    if len(rejections) >= 2:
        triggers.append("repeated_rejection")
    blocked = any(event.get("layer") == "input" for event in ev.guardrail_events)
    # The input guard's own rejection is reported as blocked_input, not as an error.
    if ev.error and not (blocked and ev.error.startswith(_INPUT_REJECTED_PREFIX)):
        # A timeout or rate limit is an infrastructure event, not a behaviour
        # regression: still harvested, but labelled so the curator can tell.
        triggers.append("transient_error" if TRANSIENT_MARK in ev.error else "run_error")
    if blocked:
        triggers.append("blocked_input")
    if any(event.get("rule") == "injection_in_chunk" for event in ev.guardrail_events):
        triggers.append("dropped_chunk")
    if ev.fallbacks:
        triggers.append("fallback")
    if ev.langfuse_errors:
        triggers.append("langfuse_error")
    return triggers


def suggest_expectations(triggers: list[str]) -> dict[str, Any]:
    """Pre-filled hints for the curator; never applied without their decision."""

    if "blocked_input" in triggers:
        return {"must_block": True}
    return {}
