"""Compiled patterns shared by the input, retrieval and output guardrails.

One module so the same injection regex protects the objective *and* the
retrieved chunks, and so tuning a false positive happens in exactly one place.
All matching is deterministic and local: no classifier, no moderation API.
"""

import re

Pattern = tuple[str, re.Pattern[str]]

# Attempts to override the system prompt or reassign the model's role.
INJECTION_PATTERNS: list[Pattern] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b"
            r".{0,20}\b(instructions?|prompts?|rules?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "system_prompt_override",
        re.compile(
            r"\b(new|updated|real|actual)\s+system\s+prompt\b|\bsystem\s*prompt\s*:",
            re.IGNORECASE,
        ),
    ),
    ("role_reassignment", re.compile(r"\byou\s+are\s+now\s+(a|an|the)\b", re.IGNORECASE)),
    (
        "developer_mode",
        re.compile(r"\b(developer|god|jailbreak|dan)\s+mode\b", re.IGNORECASE),
    ),
]

# Attempts to pull secrets or arbitrary files out through the agents.
EXFILTRATION_PATTERNS: list[Pattern] = [
    (
        "secret_request",
        re.compile(
            r"\b(api[\s_-]?keys?|passwords?|secrets?|tokens?|credentials?)\b"
            r".{0,40}\b(print|reveal|show|output|dump|leak|return|include)\b"
            r"|\b(print|reveal|show|output|dump|leak|return)\b.{0,40}"
            r"\b(api[\s_-]?keys?|passwords?|secrets?|tokens?|credentials?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    ("env_file", re.compile(r"(^|[\s'\"/])\.env\b", re.IGNORECASE)),
    (
        "code_execution",
        re.compile(
            r"\b(exec|eval|subprocess|os\.system|rm\s+-rf|curl\s+http|wget\s+http)\b",
            re.IGNORECASE,
        ),
    ),
]

# Personal data that has no business in a market report.
PII_PATTERNS: list[Pattern] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")),
    (
        "phone",
        re.compile(
            r"(?<![\w$.,])(\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.,]?\d)"
        ),
    ),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

# Credential shapes that must never reach a report or a log line.
SECRET_PATTERNS: list[Pattern] = [
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("langsmith_key", re.compile(r"\blsv2_[A-Za-z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "generic_assignment",
        re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    ),
]


def find_matches(text: str, patterns: list[Pattern]) -> list[str]:
    """Return the names of every pattern that matches `text`, in list order."""

    return [name for name, regex in patterns if regex.search(text)]
