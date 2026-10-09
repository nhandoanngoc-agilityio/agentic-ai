"""Output guardrails applied to a drafted report before the human sees it.

Three deterministic passes, in order:

1. `redact_pii`         -- emails, phone numbers, card numbers, SSN-shaped ids
2. `scrub_secrets`      -- API-key-shaped strings that must never reach disk
3. `flag_unverified_numbers` -- any figure in the draft that does not trace back
   to a research finding or an analytics result is marked `[unverified]`

Nothing here blocks: the report still reaches the approval interrupt, with
warnings attached, and the human decides. Grounding is checked by arithmetic,
not by a second model call, so this adds no tokens and no measurable latency.
"""

import calendar
import re

from market_research_team.retrieval.evidence import finding_key, superseded
from market_research_team.security.patterns import PII_PATTERNS, SECRET_PATTERNS
from market_research_team.state import AnalyticsResult, GuardrailEvent, ResearchFinding

UNVERIFIED_MARK = " [unverified]"
OUTDATED_MARK = " [outdated]"

# A numeric token as it appears in prose: optional currency, digits with
# optional thousands separators and decimals, optional magnitude suffix or %.
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.$-])(?P<currency>[$€£])?(?P<number>\d{1,3}(?:,\d{3})+|\d+)(?P<decimal>\.\d+)?"
    r"\s?(?P<suffix>[KkMmBb](?![A-Za-z])|%|percent)?(?![\w.]?\d)"
)
_MAGNITUDES = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}
_MARKDOWN_MARKER = re.compile(r"^\s*(?:#{1,6}\s|\d+[.)]\s)")
_MIN_CHECKED_VALUE = 10.0
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
OUTDATED_LABEL = re.compile(
    r"\b(outdated|previous(ly)?|earlier|formerly|old(er)?|prior|historical(ly)?|"
    r"(up|down|increased|rose|raised|grew|climbed|changed|decreased|dropped|fell|cut) from)\b",
    re.I,
)
# "from $49 to $55": a figure stated as the start of a change.
_CHANGE_RANGE = re.compile(r"\bfrom\s+\**[$€£]?\d[\d,.]*\s?[KkMmBb%]?\**\s+to\b", re.I)
_RELATIVE_TOLERANCE = 0.01


def _token_value(match: re.Match[str]) -> float:
    raw = match.group("number").replace(",", "") + (match.group("decimal") or "")
    value = float(raw)
    suffix = (match.group("suffix") or "").lower()
    return value * _MAGNITUDES.get(suffix, 1.0)


def _finding_values(findings: list[ResearchFinding]) -> set[float]:
    values: set[float] = set()
    for finding in findings:
        for match in _NUMBER_TOKEN.finditer(finding["content"]):
            values.add(_token_value(match))
    return values


def _finding_years(findings: list[ResearchFinding]) -> set[int]:
    return {int(as_of[:4]) for f in findings if (as_of := f.get("as_of"))}


def _is_source_year(match: re.Match[str], years: set[int]) -> bool:
    """A bare integer equal to a finding's year: the report citing a source's
    date ("as of 2026-03"), not a figure. Exact match only -- a year in the
    evidence set would ground every figure within 1% of it."""

    if match.group("currency") or match.group("decimal") or match.group("suffix"):
        return False
    return int(match.group("number").replace(",", "")) in years


def outdated_dates(findings: list[ResearchFinding]) -> set[str]:
    """The `as_of` dates of the findings a newer source supersedes."""

    by = superseded(findings)
    return {as_of for f in findings if finding_key(f) in by and (as_of := f.get("as_of"))}


def _date_cues(findings: list[ResearchFinding]) -> list[str]:
    """Ways a report can name a superseded source's date: `2026-03`,
    `March 2026`, `Mar 2026`, and the bare year when no current source shares it
    (`2026` alone can't tell March from August)."""

    current, outdated = _split_current(findings)
    current_years = {as_of[:4] for f in current if (as_of := f.get("as_of"))}
    cues: list[str] = []
    for as_of in {as_of for f in outdated if (as_of := f.get("as_of"))}:
        year, month = as_of[:4], int(as_of[5:7])
        cues += [
            as_of,
            f"{calendar.month_name[month]} {year}",
            f"{calendar.month_abbr[month]} {year}",
        ]
        if year not in current_years:
            cues.append(year)
    return cues


def is_labelled_outdated(sentence: str, findings: list[ResearchFinding]) -> bool:
    """Whether a sentence discloses that its figure is old -- what the reporting
    prompt asks for: an 'outdated'-type word, change wording ("up from $49",
    "from $49 to $55"), or the superseded source's date."""

    if OUTDATED_LABEL.search(sentence) or _CHANGE_RANGE.search(sentence):
        return True
    return any(re.search(rf"\b{re.escape(cue)}\b", sentence) for cue in _date_cues(findings))


def _sentence_at(line: str, position: int) -> str:
    start = 0
    for match in _SENTENCE_BREAK.finditer(line):
        if match.end() <= position:
            start = match.end()
        else:
            return line[start : match.start()]
    return line[start:]


def _evidence_values(findings: list[ResearchFinding], results: list[AnalyticsResult]) -> set[float]:
    """Numbers a report may state: those in the findings, plus the output of
    every analytics result whose own inputs came from the findings.

    The tool arguments themselves are never evidence -- the model chose them,
    so counting them would let an invented input vouch for itself (and for
    whatever was computed from it).
    """

    values = _finding_values(findings)
    for result in results:
        if not ungrounded_inputs(result.get("inputs", []), findings, evidence=values):
            values.add(float(result["value"]))
    return values


def _is_grounded(value: float, evidence: set[float]) -> bool:
    for candidate in evidence:
        if candidate == value:
            return True
        scale = max(abs(candidate), abs(value))
        if scale and abs(candidate - value) / scale <= _RELATIVE_TOLERANCE:
            return True
        # A percentage may be reported rounded to one decimal or as an integer.
        if abs(candidate - value) <= 0.05:
            return True
    return False


def ungrounded_inputs(
    inputs: list[float],
    findings: list[ResearchFinding],
    *,
    evidence: set[float] | None = None,
) -> list[float]:
    """The tool inputs (>= 10, like the report check) that match no number in
    the findings: figures the analytics model supplied rather than read."""

    known = _finding_values(findings) if evidence is None else evidence
    return [
        value
        for value in inputs
        if abs(value) >= _MIN_CHECKED_VALUE and not _is_grounded(abs(value), known)
    ]


def redact_pii(text: str) -> tuple[str, list[GuardrailEvent]]:
    """Replace personal data with a typed placeholder; one event per pattern class."""

    events: list[GuardrailEvent] = []
    for name, regex in PII_PATTERNS:
        text, count = regex.subn(f"[{name} redacted]", text)
        if count:
            events.append(
                {
                    "layer": "output",
                    "rule": f"pii_{name}",
                    "detail": f"{count} occurrence(s) redacted",
                }
            )
    return text, events


def scrub_secrets(text: str) -> tuple[str, list[GuardrailEvent]]:
    """Replace credential-shaped strings; one event per pattern class."""

    events: list[GuardrailEvent] = []
    for name, regex in SECRET_PATTERNS:
        text, count = regex.subn("[secret removed]", text)
        if count:
            events.append(
                {
                    "layer": "output",
                    "rule": f"secret_{name}",
                    "detail": f"{count} occurrence(s) removed",
                }
            )
    return text, events


def _split_current(
    findings: list[ResearchFinding],
) -> tuple[list[ResearchFinding], list[ResearchFinding]]:
    """(current, outdated) findings: outdated ones a newer source supersedes."""

    by = superseded(findings)
    current = [finding for finding in findings if finding_key(finding) not in by]
    outdated = [finding for finding in findings if finding_key(finding) in by]
    return current, outdated


def flag_unverified_numbers(
    text: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
) -> tuple[str, list[GuardrailEvent]]:
    """Mark every figure no current evidence supports: ` [outdated]` when only
    a superseded finding states it, ` [unverified]` otherwise.

    Numbers under 10 and markdown list/heading markers are skipped: they are
    almost always structure or counts of things in the report itself, and
    flagging them would train reviewers to ignore the mark.
    """

    current, outdated_findings = _split_current(findings)
    evidence = _evidence_values(current, results)
    stale = _finding_values(outdated_findings)
    years = _finding_years(findings)
    unverified: list[str] = []
    outdated: list[str] = []

    def _annotate_line(line: str) -> str:
        prefix_match = _MARKDOWN_MARKER.match(line)
        prefix_end = prefix_match.end() if prefix_match else 0

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if match.start() < prefix_end:
                return token
            value = _token_value(match)
            if value < _MIN_CHECKED_VALUE or _is_grounded(value, evidence):
                return token
            if _is_source_year(match, years):
                return token
            if _is_grounded(value, stale):
                if is_labelled_outdated(_sentence_at(line, match.start()), findings):
                    return token  # the report already says it's old
                outdated.append(token.strip())
                return token + OUTDATED_MARK
            unverified.append(token.strip())
            return token + UNVERIFIED_MARK

        return _NUMBER_TOKEN.sub(_replace, line)

    annotated = "\n".join(_annotate_line(line) for line in text.split("\n"))
    events: list[GuardrailEvent] = []
    if unverified:
        events.append(
            {
                "layer": "output",
                "rule": "unverified_numbers",
                "detail": f"{len(unverified)} figure(s) not traceable to evidence: "
                + ", ".join(unverified[:10]),
            }
        )
    if outdated:
        events.append(
            {
                "layer": "output",
                "rule": "stale_figure",
                "detail": f"{len(outdated)} figure(s) only an outdated source supports: "
                + ", ".join(outdated[:10]),
            }
        )
    return annotated, events


def figures_only_outdated_sources_support(
    text: str, findings: list[ResearchFinding]
) -> list[tuple[str, str]]:
    """(figure, sentence) for each figure in `text` that only superseded
    findings support -- for checks that look at how a sentence labels it."""

    current, outdated_findings = _split_current(findings)
    evidence, stale = _finding_values(current), _finding_values(outdated_findings)
    years = _finding_years(findings)
    found: list[tuple[str, str]] = []
    for sentence in _SENTENCE.split(text):
        for match in _NUMBER_TOKEN.finditer(sentence):
            value = _token_value(match)
            if value < _MIN_CHECKED_VALUE or _is_grounded(value, evidence):
                continue
            if _is_source_year(match, years):
                continue
            if _is_grounded(value, stale):
                found.append((match.group(0).strip(), sentence.strip()))
    return found


def apply_output_guardrails(
    text: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
) -> tuple[str, list[GuardrailEvent]]:
    """Run every output pass and collect their events."""

    text, pii_events = redact_pii(text)
    text, secret_events = scrub_secrets(text)
    text, number_events = flag_unverified_numbers(text, findings, results)
    return text, pii_events + secret_events + number_events


def warnings_from_events(events: list[GuardrailEvent]) -> list[str]:
    """Human-readable one-liners for the approval prompt."""

    return [f"{event['rule']}: {event['detail']}" for event in events]
