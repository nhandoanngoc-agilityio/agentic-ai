"""Long-term memory: reviewer feedback remembered across runs.

When a reviewer rejects a draft with feedback ("add a pricing table", "cover
both vendors"), the note is kept in the LangGraph store and later drafts see
the most recent ones as guidance. Per-run state lives in the checkpointer;
this is what outlives a run.

Stored notes reach future prompts, so they are treated as untrusted on the
way in: capped in length, PII and secrets redacted, and a note that reads like
a prompt injection is refused rather than stored.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.store.base import BaseStore

from market_research_team.config import settings
from market_research_team.security.output_filters import redact_pii, scrub_secrets
from market_research_team.security.patterns import INJECTION_PATTERNS, find_matches
from market_research_team.state import GuardrailEvent

NAMESPACE = ("reviewer_feedback",)
_MAX_NOTE_CHARS = 500
# Search returns this many candidates to sort by recency (the store's own
# order is not guaranteed); far more than the notes ever shown.
_SEARCH_LIMIT = 200


def current_store() -> BaseStore | None:
    """The store the running graph was compiled with, or None (evals, tests,
    or outside a graph run)."""

    try:
        from langgraph.config import get_store

        return get_store()
    except Exception:
        return None


def remember_feedback(
    store: BaseStore | None, feedback: str, *, objective: str, thread_id: str | None
) -> GuardrailEvent | None:
    """Store a reviewer's note for later drafts. Returns a guardrail event when
    the note was refused (it reads like a prompt injection), else None."""

    if store is None or settings.run_policy.max_reviewer_notes <= 0:
        return None
    note = " ".join(feedback.split())
    if not note:
        return None
    hits = find_matches(note, INJECTION_PATTERNS)
    if hits:
        return {
            "layer": "input",
            "rule": "reviewer_note_refused",
            "detail": f"reviewer feedback not remembered: {', '.join(hits)}",
        }
    # Redact first, cut after: cut first, a key split at the limit is too short
    # for the secret patterns and its prefix would be stored.
    note, _ = redact_pii(note)
    note, _ = scrub_secrets(note)
    note = note[:_MAX_NOTE_CHARS]
    store.put(
        NAMESPACE,
        uuid.uuid4().hex,
        {
            "feedback": note,
            "objective": objective,
            "thread_id": thread_id,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return None


def recent_feedback(store: BaseStore | None, *, exclude_thread: str | None) -> list[str]:
    """The newest reviewer notes from other runs, within the retention window."""

    limit = settings.run_policy.max_reviewer_notes
    if store is None or limit <= 0:
        return []
    cutoff = datetime.now(UTC) - timedelta(days=settings.audit_retention_days)
    values: list[dict[str, Any]] = [
        item.value for item in store.search(NAMESPACE, limit=_SEARCH_LIMIT)
    ]
    fresh = [
        value
        for value in values
        if value.get("thread_id") != exclude_thread
        and datetime.fromisoformat(value["created_at"]) >= cutoff
    ]
    fresh.sort(key=lambda value: value["created_at"], reverse=True)
    notes: list[str] = []
    for value in fresh:
        if value["feedback"] not in notes:
            notes.append(value["feedback"])
        if len(notes) == limit:
            break
    return notes


def prune_memory(store: BaseStore, cutoff: datetime, *, apply: bool) -> int:
    """Delete reviewer notes created before `cutoff` (retention, see
    feedback/retention.py). Returns how many were (or would be) deleted."""

    old = [
        item
        for item in store.search(NAMESPACE, limit=10_000)
        if datetime.fromisoformat(item.value["created_at"]) < cutoff
    ]
    if apply:
        for item in old:
            store.delete(NAMESPACE, item.key)
    return len(old)
