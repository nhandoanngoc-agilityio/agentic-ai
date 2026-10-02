"""Review queue of harvested production failures, waiting for a curator.

Files live under `settings.candidates_dir` (git-ignored: they hold production
text): `pending/<id>.json` until a person promotes or rejects them, then
`promoted/` or `rejected/`. Nothing is ever deleted. Every write is atomic
(temp file + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from market_research_team.feedback.regressions import (
    Expectations,
    RegressionEntry,
    append_regression,
    load_regressions,
)
from market_research_team.feedback.signals import RunEvidence
from market_research_team.feedback.triggers import suggest_expectations
from market_research_team.security.patterns import PII_PATTERNS, SECRET_PATTERNS, find_matches

_SENSITIVE = PII_PATTERNS + SECRET_PATTERNS
_STATES = ("pending", "promoted", "rejected")


class PromotionError(ValueError):
    """A candidate cannot become a regression case yet; the message says why."""


def scrub_text(text: str) -> str:
    for _, regex in _SENSITIVE:
        text = regex.sub("[redacted]", text)
    return text


def candidate_id(objective: str) -> str:
    """Hash of the scrubbed, normalised objective: the committed id never
    derives from personal data, and objectives differing only in it merge."""

    normalised = re.sub(r"\s+", " ", scrub_text(objective).strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def _evidence_record(ev: RunEvidence, triggers: list[str]) -> dict[str, Any]:
    decisions = [
        {**d, "feedback": scrub_text(str(d["feedback"]))} if d.get("feedback") else dict(d)
        for d in ev.decisions
    ]
    return {
        "thread_id": ev.thread_id,
        "agent_version": ev.agent_version,
        "first_seen": ev.first_seen,
        "last_seen": ev.last_seen,
        "triggers": triggers,
        "route_trace": ev.route_trace,
        "guardrail_events": [
            {**g, "detail": scrub_text(str(g.get("detail", "")))} for g in ev.guardrail_events
        ],
        "error": scrub_text(ev.error) if ev.error else None,
        "report_path": ev.report_path,
        "decisions": decisions,
        "ratings": ev.ratings,
        "fallbacks": ev.fallbacks,
        "langfuse_trace_ids": ev.langfuse_trace_ids,
        "langfuse_errors": [scrub_text(e) for e in ev.langfuse_errors],
        "sources": ev.sources,
    }


@dataclass
class Candidate:
    id: str
    objective: str
    triggers: list[str]
    occurrences: int
    evidence: list[dict[str, Any]]
    suggested: dict[str, Any] = field(default_factory=dict)
    expectations: Expectations = field(default_factory=Expectations)
    note: str = ""
    reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "triggers": self.triggers,
            "occurrences": self.occurrences,
            "evidence": self.evidence,
            "suggested": self.suggested,
            "expectations": self.expectations.to_dict(),
            "note": self.note,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(
            id=str(data["id"]),
            objective=str(data["objective"]),
            triggers=list(data.get("triggers", [])),
            occurrences=int(data.get("occurrences", 1)),
            evidence=list(data.get("evidence", [])),
            suggested=dict(data.get("suggested", {})),
            expectations=Expectations.from_dict(data.get("expectations", {})),
            note=str(data.get("note", "")),
            reject_reason=str(data.get("reject_reason", "")),
        )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class CandidateQueue:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, state: str, cid: str) -> Path:
        return self.root / state / f"{cid}.json"

    def _state_of(self, cid: str) -> str | None:
        return next((s for s in _STATES if self._path(s, cid).exists()), None)

    def save(self, candidate: Candidate, state: str = "pending") -> None:
        _atomic_write(
            self._path(state, candidate.id), json.dumps(candidate.to_dict(), indent=2) + "\n"
        )

    def load(self, cid: str) -> Candidate:
        path = self._path("pending", cid)
        if not path.exists():
            raise KeyError(f"No pending candidate {cid!r}")
        return Candidate.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_pending(self) -> tuple[list[Candidate], list[str]]:
        folder = self.root / "pending"
        if not folder.exists():
            return [], []
        valid, invalid = [], []
        for path in sorted(folder.glob("*.json")):
            try:
                valid.append(Candidate.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid.append(path.stem)
        return valid, invalid

    def add(self, ev: RunEvidence, triggers: list[str]) -> str:
        """Queue a failing run. Returns "new", "merged" (same objective already
        pending), "skipped" (already promoted or rejected) or "invalid" (the
        pending file for this id cannot be read; it is left untouched)."""

        cid = candidate_id(ev.objective)
        state = self._state_of(cid)
        if state in ("promoted", "rejected"):
            return "skipped"
        record = _evidence_record(ev, triggers)
        if state == "pending":
            try:
                candidate = self.load(cid)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return "invalid"
            if all(e["thread_id"] != ev.thread_id for e in candidate.evidence):
                candidate.evidence.append(record)
                candidate.occurrences += 1
            for trigger in triggers:
                if trigger not in candidate.triggers:
                    candidate.triggers.append(trigger)
            candidate.suggested = suggest_expectations(candidate.triggers)
            self.save(candidate)
            return "merged"
        self.save(
            Candidate(
                id=cid,
                objective=scrub_text(ev.objective),
                triggers=list(triggers),
                occurrences=1,
                evidence=[record],
                suggested=suggest_expectations(triggers),
            )
        )
        return "new"

    def _validate(self, c: Candidate, existing: set[str]) -> None:
        e = c.expectations
        if e.is_empty():
            raise PromotionError("Set at least one expectation before promoting.")
        if e.must_block and e != Expectations(must_block=True):
            raise PromotionError("must_block cannot be combined with other expectations.")
        texts = [c.objective, c.note, *e.required_facts, *e.forbidden_substrings]
        if any(find_matches(text, _SENSITIVE) for text in texts):
            raise PromotionError(
                "The objective, note or an expectation still contains personal data or a secret."
            )
        if c.id in existing:
            raise PromotionError(f"Case {c.id} is already in the regressions file.")

    def promote(self, cid: str, regressions_path: Path, *, now: str) -> RegressionEntry:
        candidate = self.load(cid)
        existing = {entry.id for entry in load_regressions(regressions_path)}
        self._validate(candidate, existing)
        entry = RegressionEntry(
            id=candidate.id,
            objective=candidate.objective,
            expectations=candidate.expectations,
            origin={
                "triggers": candidate.triggers,
                "harvested_at": now,
                "agent_versions": sorted(
                    {e["agent_version"] for e in candidate.evidence if e.get("agent_version")}
                ),
                "note": candidate.note,
            },
        )
        append_regression(regressions_path, entry)
        self.save(candidate, "promoted")
        self._path("pending", cid).unlink()
        return entry

    def reject(self, cid: str, reason: str) -> None:
        candidate = self.load(cid)
        candidate.reject_reason = reason
        self.save(candidate, "rejected")
        self._path("pending", cid).unlink()

    def last_harvest(self) -> str | None:
        marker = self.root / ".last_harvest"
        return marker.read_text(encoding="utf-8").strip() if marker.exists() else None

    def set_last_harvest(self, ts: str) -> None:
        _atomic_write(self.root / ".last_harvest", ts + "\n")
