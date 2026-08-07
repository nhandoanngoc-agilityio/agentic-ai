"""Reusable, deterministic check functions for prompt evaluation cases.

Every check takes concrete outputs (not an LLM) and returns
`(passed, detail)`, so the scoring logic itself stays pure and
unit-testable without needing real LLM calls — only the eval cases that
call these need real credentials.
"""

from market_research_team.state import AnalyticsResult


def check_query_count(queries: list[str], *, min_count: int, max_count: int) -> tuple[bool, str]:
    count = len(queries)
    passed = min_count <= count <= max_count
    return passed, f"got {count} queries (expected {min_count}-{max_count})"


def check_keyword_coverage(text_items: list[str], required_any: list[str]) -> tuple[bool, str]:
    """Passes if at least one of `required_any` appears in at least one item (case-insensitive)."""

    if not required_any:
        return True, "no keywords required"

    haystack = " ".join(text_items).lower()
    hits = [keyword for keyword in required_any if keyword.lower() in haystack]
    passed = len(hits) > 0
    return passed, f"matched keywords: {hits or 'none'} (looked for any of {required_any})"


def check_contains_all(text: str, required_substrings: list[str]) -> tuple[bool, str]:
    if not required_substrings:
        return True, "nothing required"

    lowered = text.lower()
    missing = [item for item in required_substrings if item.lower() not in lowered]
    passed = not missing
    return passed, ("all present" if passed else f"missing: {missing}")


def check_min_length(items: list, minimum: int, label: str) -> tuple[bool, str]:
    passed = len(items) >= minimum
    return passed, f"{label} count={len(items)} (expected >= {minimum})"


def check_any_value_matches(
    results: list[AnalyticsResult], plausible_values: list[float], tolerance: float
) -> tuple[bool, str]:
    """Passes if any result's value is close to any plausible grounded value.

    Catches two failure modes at once: no tool called at all (empty
    results), and a tool called with a hallucinated number instead of one
    actually derivable from the findings.
    """

    if not plausible_values:
        return True, "no specific value check requested"

    matched = [
        result
        for result in results
        if any(abs(result["value"] - plausible) <= tolerance for plausible in plausible_values)
    ]
    passed = len(matched) > 0
    detail = f"matched results: {matched or 'none'} against plausible values {plausible_values}"
    return passed, detail
