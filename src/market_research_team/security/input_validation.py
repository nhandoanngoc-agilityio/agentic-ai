"""Input validation for text that becomes graph state, applied once at the graph boundary.

Deterministic and local on purpose: length bounds, control-character
stripping and a small pattern denylist for prompt-injection and secret
exfiltration attempts. No moderation API, no classifier -- objectives are
short analyst questions, and a network round-trip per run would cost more
latency and false positives than it would buy.
"""

import re

from market_research_team.security.patterns import (
    EXFILTRATION_PATTERNS,
    INJECTION_PATTERNS,
    find_matches,
)

_MAX_OBJECTIVE_LENGTH = 2000
_MIN_OBJECTIVE_LENGTH = 8

# C0 control characters except tab (0x09) and newline (0x0A); carriage returns
# are normalised to newlines before this is applied.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def validate_objective(text: str) -> str:
    """Validate and normalize a research objective before it enters graph state.

    Raises ValueError if the objective is empty/whitespace-only, too short,
    exceeds the maximum length, or matches a prompt-injection or secret
    exfiltration pattern; returns the cleaned string otherwise.
    """

    cleaned = _CONTROL_CHARS.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not cleaned:
        raise ValueError("Research objective must not be empty.")
    if len(cleaned) > _MAX_OBJECTIVE_LENGTH:
        raise ValueError(
            f"Research objective exceeds maximum length of {_MAX_OBJECTIVE_LENGTH} characters."
        )
    if len(cleaned) < _MIN_OBJECTIVE_LENGTH:
        raise ValueError(
            f"Research objective is too short (minimum {_MIN_OBJECTIVE_LENGTH} characters)."
        )

    injection_hits = find_matches(cleaned, INJECTION_PATTERNS)
    if injection_hits:
        raise ValueError(
            f"Research objective rejected: prompt-injection pattern ({', '.join(injection_hits)})."
        )
    exfiltration_hits = find_matches(cleaned, EXFILTRATION_PATTERNS)
    if exfiltration_hits:
        raise ValueError(
            f"Research objective rejected: out-of-scope request ({', '.join(exfiltration_hits)})."
        )
    return cleaned
