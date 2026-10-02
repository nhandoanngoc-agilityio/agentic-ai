"""Committed regression cases: one JSON object per line in evals/regressions.jsonl.

Each line came from a curated production failure (see candidates.py). Adding
a line is a reviewed change -- the "add eval" step of the feedback loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Expectations:
    must_block: bool = False
    required_facts: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    min_findings: int | None = None
    requires_approval: bool = False

    def is_empty(self) -> bool:
        return not (
            self.must_block
            or self.required_facts
            or self.forbidden_substrings
            or self.min_findings is not None
            or self.requires_approval
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expectations:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class RegressionEntry:
    id: str
    objective: str
    expectations: Expectations
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "expectations": self.expectations.to_dict(),
            "origin": self.origin,
        }


def load_regressions(path: Path) -> list[RegressionEntry]:
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            entries.append(
                RegressionEntry(
                    id=str(raw["id"]),
                    objective=str(raw["objective"]),
                    expectations=Expectations.from_dict(raw.get("expectations", {})),
                    origin=dict(raw.get("origin", {})),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{path}: line {number} is not a valid regression case: {exc}"
            ) from exc
    return entries


def append_regression(path: Path, entry: RegressionEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
