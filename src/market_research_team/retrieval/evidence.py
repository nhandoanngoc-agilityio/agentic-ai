"""Provenance of retrieved evidence: topic, entity, and which findings a
newer source supersedes.

A finding is superseded when another finding about the same entity and
topic, dated later, states figures of its own. Both need an `as_of`, an
entity and a topic other than `other`; anything missing means "never
superseded", so incomplete metadata can hide a warning but never a finding.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from market_research_team.state import ResearchFinding

# (topic, heading pattern). Order matters: the first match wins. Whole-word
# patterns, so "Planning Horizon" doesn't read as pricing.
TOPIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("pricing", r"\b(pricing|prices?|tiers?|plans?|costs?|contract value)\b"),
    ("customers", r"\b(customers?|clients?)\b"),
    ("headcount", r"\b(headcount|employees?)\b"),
    ("market_size", r"\b(market size|growth|forecasts?)\b"),
    ("strengths", r"\bstrengths?\b"),
    ("weaknesses", r"\bweaknesses?\b"),
    ("recent_moves", r"\b(recent|moves|news)\b"),
)
_TOPIC_PATTERNS = [(topic, re.compile(pattern, re.IGNORECASE)) for topic, pattern in TOPIC_KEYWORDS]
_DIGIT = re.compile(r"\d")


def headings_of(metadata: Mapping[str, Any]) -> list[str]:
    """The chunk's markdown headings, outermost first (`h1`, `h2`, `h3`)."""

    return [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]


def topic_for(headings: list[str]) -> str:
    """The topic of the deepest heading that names one, else `other`."""

    for heading in reversed(headings):
        for topic, pattern in _TOPIC_PATTERNS:
            if pattern.search(heading):
                return topic
    return "other"


def resolve_entity(doc_entity: str | None, headings: list[str], known: Iterable[str]) -> str | None:
    """The chunk's entity: the document's, or for a `Multiple` document the
    deepest heading naming a known competitor (case-insensitive), else
    `Market`."""

    if doc_entity is None:
        return None
    if doc_entity.lower() != "multiple":
        return doc_entity
    by_lower = {name.lower(): name for name in known}
    for heading in reversed(headings):
        name = by_lower.get(heading.strip().lower())
        if name is not None:
            return name
    return "Market"


def finding_key(finding: ResearchFinding) -> tuple[str, str]:
    return finding["source"], finding["content"]


def _scope(finding: ResearchFinding) -> tuple[str, str] | None:
    entity, topic = finding.get("entity"), finding.get("topic")
    if not entity or not topic or topic == "other" or not finding.get("as_of"):
        return None
    return entity.lower(), topic


def superseded(findings: list[ResearchFinding]) -> dict[tuple[str, str], str]:
    """`finding_key` -> source of the newest finding that supersedes it.

    Only a finding that states a figure (any digit) can supersede: a newer
    section that merely talks about pricing must not hide an older price.
    Equal dates supersede nothing -- there's no honest tie-break.
    """

    newest: dict[tuple[str, str], tuple[str, str]] = {}
    for finding in findings:
        scope = _scope(finding)
        if scope is None or not _DIGIT.search(finding["content"]):
            continue
        as_of = finding.get("as_of", "")
        if scope not in newest or as_of > newest[scope][0]:
            newest[scope] = (as_of, finding["source"])
    result: dict[tuple[str, str], str] = {}
    for finding in findings:
        scope = _scope(finding)
        if scope is not None and scope in newest:
            as_of, source = newest[scope]
            if finding.get("as_of", "") < as_of:
                result[finding_key(finding)] = source
    return result


def finding_provenance(
    finding: ResearchFinding, superseded_by: Mapping[tuple[str, str], str]
) -> str:
    """`entity · topic · as of YYYY-MM`, plus `(superseded by …)`; empty for
    an undated finding, which keeps every prompt as it was before dates."""

    as_of = finding.get("as_of")
    if not as_of:
        return ""
    parts = [value for value in (finding.get("entity"), finding.get("topic")) if value]
    parts.append(f"as of {as_of}")
    provenance = " · ".join(parts)
    newer = superseded_by.get(finding_key(finding))
    return f"{provenance} (superseded by {newer})" if newer else provenance


def finding_label(finding: ResearchFinding, superseded_by: Mapping[tuple[str, str], str]) -> str:
    """`source · entity · topic · as of YYYY-MM (superseded by …)`, or the
    source alone for an undated finding."""

    provenance = finding_provenance(finding, superseded_by)
    return f"{finding['source']} · {provenance}" if provenance else finding["source"]
