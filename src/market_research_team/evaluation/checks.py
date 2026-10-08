"""Reusable, deterministic check functions for prompt evaluation cases.

Every check takes concrete outputs (not an LLM) and returns
`(passed, detail)`, so the scoring logic itself stays pure and
unit-testable without needing real LLM calls — only the eval cases that
call these need real credentials.
"""

import re
from typing import Any

from market_research_team.state import AnalyticsResult, PlanItem, ResearchFinding


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


def check_source_coverage(
    findings: list[ResearchFinding], expected_sources: list[str], min_hits: int
) -> tuple[bool, str]:
    """Passes if at least `min_hits` of `expected_sources` appear among the
    retrieved/reranked findings' `source` field.

    Catches a retrieval/reranking regression that a query-count or keyword
    check on the rewritten queries alone would miss: the right queries were
    generated, but the wrong (or no) chunks made it through retrieval and
    reranking.
    """

    if not expected_sources:
        return True, "no sources required"

    retrieved_sources = {finding["source"] for finding in findings}
    hits = [source for source in expected_sources if source in retrieved_sources]
    passed = len(hits) >= min_hits
    detail = (
        f"matched sources: {hits or 'none'} (expected >= {min_hits} of {expected_sources}; "
        f"retrieved: {sorted(retrieved_sources)})"
    )
    return passed, detail


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


def check_supervisor_route(
    decision: str,
    focus: str | None,
    expected: tuple[str, ...],
    focus_keywords: list[str],
    decided_by: str = "llm",
) -> tuple[bool, str]:
    """Passes if the decision is one of `expected`, was not a fallback (the
    model's own choice was invalid or its call failed), and, for a hand-back
    to Research, the focus mentions every keyword (case-insensitive)."""

    decision_ok = decision in expected and decided_by != "fallback"
    missing = [kw for kw in focus_keywords if kw.lower() not in (focus or "").lower()]
    focus_ok = decision != "research" or not missing
    detail = f"decision={decision!r} by {decided_by} (expected: {expected}); focus={focus!r}"
    if missing and decision == "research":
        detail += f"; focus missing {missing}"
    return decision_ok and focus_ok, detail


_COMPARISON = re.compile(r"\b(compar\w*|differ\w*|versus|vs\.?|relative to|than)\b", re.IGNORECASE)


def check_plan(
    questions: list[str], *, min_items: int, max_items: int, must_mention: list[str]
) -> tuple[bool, str]:
    """Passes if the plan has `min_items`..`max_items` sub-questions, every
    `must_mention` word appears in some question, and no question restates
    the comparison (names several of them and is phrased as a comparison): no
    document compares companies, so such a question can't be answered. A
    question that only mentions several companies as context (e.g. the market
    they operate in) is fine."""

    lowered = [question.lower() for question in questions]
    count_ok = min_items <= len(questions) <= max_items
    uncovered = [word for word in must_mention if not any(word.lower() in q for q in lowered)]
    words = [word.lower() for word in must_mention]
    restated = [
        question
        for question, low in zip(questions, lowered, strict=True)
        if sum(word in low for word in words) > 1 and _COMPARISON.search(question)
    ]
    passed = count_ok and not uncovered and not restated
    return passed, (
        f"{len(questions)} item(s) (expected {min_items}-{max_items}); "
        f"uncovered: {uncovered or 'none'}; restated comparisons: {restated or 'none'}"
    )


def check_inputs_grounded(
    results: list[AnalyticsResult], findings: list[ResearchFinding]
) -> tuple[bool, str]:
    """Passes if every numeric tool input appears in the findings: the model
    computed from figures it read, not figures it supplied."""

    from market_research_team.security.output_filters import ungrounded_inputs

    invented = {
        result["metric"]: missing
        for result in results
        if (missing := ungrounded_inputs(result.get("inputs", []), findings))
    }
    return not invented, f"ungrounded tool inputs: {invented or 'none'}"


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)


def check_markdown_table(text: str) -> tuple[bool, str]:
    """Passes if `text` contains a markdown table (a header separator row)."""

    found = bool(_TABLE_SEPARATOR.search(text))
    return found, "markdown table present" if found else "no markdown table"


def check_trajectory(
    plan: list[PlanItem],
    decisions: list[dict[str, Any]],
    report_text: str,
) -> tuple[bool, str]:
    """Structural properties of one full run, from its plan, its
    `route_decision` audit events and its report:

    - a plan was made, and no route fell back (a failed or invalid model call);
    - the visit cap never had to force progress (a run that hits it looped);
    - every model-chosen hand-back to Research named a focus, and no
      sub-question was targeted twice;
    - Research ran at most once per plan item plus the broad first pass;
    - every item left unanswered is named in the report's open questions.
    """

    problems: list[str] = []
    if not plan:
        problems.append("no plan")
    fallbacks = [d for d in decisions if d.get("decided_by") == "fallback"]
    if fallbacks:
        problems.append(f"{len(fallbacks)} fallback route(s)")
    unfocused = [
        d
        for d in decisions
        if d.get("decided_by") == "llm" and d.get("next") == "research" and not d.get("focus")
    ]
    if unfocused:
        problems.append(f"{len(unfocused)} hand-back(s) without a focus")
    if any("visit cap" in str(d.get("rationale", "")) for d in decisions):
        problems.append("the visit cap forced progress")
    focuses = [str(d["focus"]) for d in decisions if d.get("next") == "research" and d.get("focus")]
    repeated = sorted({focus for focus in focuses if focuses.count(focus) > 1})
    if repeated:
        problems.append(f"sub-question(s) targeted more than once: {repeated}")
    research_passes = sum(1 for d in decisions if d.get("next") == "research")
    if research_passes > len(plan) + 1:
        problems.append(f"{research_passes} research passes for {len(plan)} plan item(s)")
    unstated = [
        item["question"]
        for item in plan
        if item["status"] != "answered" and item["question"] not in report_text
    ]
    if unstated:
        problems.append(f"unanswered but not in the report: {unstated}")

    route = " -> ".join(f"{d.get('next')}({d.get('decided_by')})" for d in decisions)
    return not problems, f"route: {route}; " + ("; ".join(problems) or "ok")
