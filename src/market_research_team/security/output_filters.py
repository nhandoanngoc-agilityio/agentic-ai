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

import re

from market_research_team.security.patterns import PII_PATTERNS, SECRET_PATTERNS
from market_research_team.state import AnalyticsResult, GuardrailEvent, ResearchFinding

UNVERIFIED_MARK = " [unverified]"

# A numeric token as it appears in prose: optional currency, digits with
# optional thousands separators and decimals, optional magnitude suffix or %.
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.$-])(?P<currency>[$€£])?(?P<number>\d{1,3}(?:,\d{3})+|\d+)(?P<decimal>\.\d+)?"
    r"\s?(?P<suffix>[KkMmBb](?![A-Za-z])|%|percent)?(?![\w.]?\d)"
)
_MAGNITUDES = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}
_MARKDOWN_MARKER = re.compile(r"^\s*(?:#{1,6}\s|\d+[.)]\s)")
_MIN_CHECKED_VALUE = 10.0
_RELATIVE_TOLERANCE = 0.01


def _token_value(match: re.Match[str]) -> float:
    raw = match.group("number").replace(",", "") + (match.group("decimal") or "")
    value = float(raw)
    suffix = (match.group("suffix") or "").lower()
    return value * _MAGNITUDES.get(suffix, 1.0)


def _evidence_values(findings: list[ResearchFinding], results: list[AnalyticsResult]) -> set[float]:
    values: set[float] = set()
    for finding in findings:
        for match in _NUMBER_TOKEN.finditer(finding["content"]):
            values.add(_token_value(match))
    for result in results:
        values.add(float(result["value"]))
        for match in _NUMBER_TOKEN.finditer(result["detail"]):
            values.add(_token_value(match))
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


def flag_unverified_numbers(
    text: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
) -> tuple[str, list[GuardrailEvent]]:
    """Append `[unverified]` to every figure with no matching evidence value.

    Numbers under 10 and markdown list/heading markers are skipped: they are
    almost always structure or counts of things in the report itself, and
    flagging them would train reviewers to ignore the mark.
    """

    evidence = _evidence_values(findings, results)
    unverified: list[str] = []

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
    return annotated, events


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
