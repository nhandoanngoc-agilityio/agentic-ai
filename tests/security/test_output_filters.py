from typing import Any

import pytest

from market_research_team.security.output_filters import (
    OUTDATED_MARK,
    UNVERIFIED_MARK,
    apply_output_guardrails,
    figures_only_outdated_sources_support,
    flag_unverified_numbers,
    is_labelled_outdated,
    redact_pii,
    scrub_secrets,
    ungrounded_inputs,
    warnings_from_events,
)
from market_research_team.state import AnalyticsResult, ResearchFinding


def _findings() -> list[ResearchFinding]:
    return [
        {
            "source": "competitor_globex.md",
            "content": (
                "Industry estimates place typical annual contract value between $150K and "
                "$400K. Globex serves around 400 large accounts with 900 employees."
            ),
            "relevance_score": 1.0,
        },
        {
            "source": "competitor_acme.md",
            "content": "Acme reports roughly 1,200 paying customers and was founded in 2015.",
            "relevance_score": 0.9,
        },
    ]


def _results() -> list[AnalyticsResult]:
    return [
        {"metric": "mean_acv", "value": 275000.0, "detail": "mean of 150000 and 400000"},
        {"metric": "headcount_pct_change", "value": 164.70588, "detail": "340 -> 900"},
    ]


def test_redact_pii_replaces_email_and_phone() -> None:
    text = "Contact jane.doe@acme.com or call 512-555-0199 for pricing."
    redacted, events = redact_pii(text)
    assert "jane.doe@acme.com" not in redacted
    assert "512-555-0199" not in redacted
    assert {event["rule"] for event in events} == {"pii_email", "pii_phone"}
    assert all(event["layer"] == "output" for event in events)


def test_redact_pii_leaves_business_figures_alone() -> None:
    text = "ACV between $150,000 and $400,000; 1,200 customers; founded 2015."
    redacted, events = redact_pii(text)
    assert redacted == text
    assert events == []


def test_scrub_secrets_removes_key_shaped_strings() -> None:
    text = "Config used sk-ant-abcdefghijklmnopqrstuvwxyz0123 and lsv2_pt_abcdefghijklmnopqrstuv."
    scrubbed, events = scrub_secrets(text)
    assert "sk-ant-" not in scrubbed
    assert "lsv2_" not in scrubbed
    assert {event["rule"] for event in events} == {"secret_anthropic_key", "secret_langsmith_key"}


def test_flag_unverified_numbers_keeps_grounded_figures_unmarked() -> None:
    draft = (
        "# Report\n"
        "- Globex ACV ranges from $150K to $400K, mean ACV ≈ $275,000.\n"
        "- Acme has 1,200 customers; Globex has 400 accounts and 900 employees.\n"
        "- Globex headcount is 164.7% larger. Acme was founded in 2015.\n"
    )
    annotated, events = flag_unverified_numbers(draft, _findings(), _results())
    assert UNVERIFIED_MARK not in annotated
    assert events == []


def test_flag_unverified_numbers_marks_invented_figures() -> None:
    draft = "Acme's revenue reached $85M last year across 1,200 customers."
    annotated, events = flag_unverified_numbers(draft, _findings(), _results())
    assert f"$85M{UNVERIFIED_MARK}" in annotated
    assert f"1,200{UNVERIFIED_MARK}" not in annotated
    assert len(events) == 1
    assert events[0]["rule"] == "unverified_numbers"
    assert "$85M" in events[0]["detail"]


def test_flag_unverified_numbers_skips_markdown_markers_and_small_numbers() -> None:
    draft = "## 2 Key Findings\n1. Three vendors\n2. Acme has 3 funding rounds and 1,200 customers"
    annotated, events = flag_unverified_numbers(draft, _findings(), _results())
    assert annotated == draft
    assert events == []


def test_apply_output_guardrails_runs_all_passes_and_collects_events() -> None:
    draft = "Email ceo@acme.com. Revenue $85M. Customers 1,200."
    cleaned, events = apply_output_guardrails(draft, _findings(), _results())
    assert "ceo@acme.com" not in cleaned
    assert f"$85M{UNVERIFIED_MARK}" in cleaned
    rules = [event["rule"] for event in events]
    assert rules == ["pii_email", "unverified_numbers"]
    assert warnings_from_events(events)[0].startswith("pii_email:")


def test_a_result_computed_from_invented_inputs_is_not_evidence() -> None:
    """The model chose the tool arguments, so they can't vouch for themselves:
    mean([120, 180]) = 150 grounds neither 120, 180 nor 150."""

    invented: list[AnalyticsResult] = [
        {
            "metric": "mean",
            "value": 150.0,
            "detail": "mean([120, 180]) = 150",
            "entity": None,
            "inputs": [120.0, 180.0],
        }
    ]
    draft = "Average deal size is $150 across deals of $120 and $180."

    annotated, events = flag_unverified_numbers(draft, _findings(), invented)

    assert annotated.count(UNVERIFIED_MARK.strip()) == 3
    assert len(events) == 1
    assert all(figure in events[0]["detail"] for figure in ("$150", "$120", "$180"))


def test_a_result_computed_from_grounded_inputs_is_evidence() -> None:
    grounded: list[AnalyticsResult] = [
        {
            "metric": "mean",
            "value": 275000.0,
            "detail": "mean",
            "entity": None,
            "inputs": [150000.0, 400000.0],
        }
    ]

    annotated, events = flag_unverified_numbers("Mean ACV is $275,000.", _findings(), grounded)

    assert UNVERIFIED_MARK not in annotated
    assert events == []


def test_ungrounded_inputs_ignores_small_numbers_and_tolerates_rounding() -> None:
    # 3 (periods) is under the checked floor; 400,001 is within 1% of 400K; 85M is invented.
    assert ungrounded_inputs([3.0, 400001.0, 85_000_000.0], _findings()) == [85_000_000.0]


_OLD: Any = {
    "source": "acme.md",
    "content": "Starter $49 per seat, 14-day trial.",
    "relevance_score": 1.0,
    "entity": "Acme",
    "topic": "pricing",
    "as_of": "2026-03",
}
_NEW: Any = {**_OLD, "source": "bench.md", "content": "Starter $55 per seat, 14-day trial."}
_NEW["as_of"] = "2026-08"


def test_a_figure_only_an_outdated_source_supports_is_marked() -> None:
    annotated, events = flag_unverified_numbers("Acme charges $49 per seat.", [_OLD, _NEW], [])

    assert OUTDATED_MARK.strip() in annotated
    assert UNVERIFIED_MARK.strip() not in annotated
    assert [e["rule"] for e in events] == ["stale_figure"]


def test_a_figure_in_both_a_current_and_an_outdated_source_is_current() -> None:
    annotated, events = flag_unverified_numbers("A 14-day trial at $55.", [_OLD, _NEW], [])

    assert annotated == "A 14-day trial at $55."
    assert events == []


def test_undated_findings_never_produce_outdated_marks() -> None:
    old = {k: v for k, v in _OLD.items() if k != "as_of"}

    annotated, events = flag_unverified_numbers("$49 and $55.", [old, _NEW], [])  # type: ignore[list-item]

    assert OUTDATED_MARK not in annotated
    assert events == []


def test_a_result_computed_from_an_outdated_input_is_not_evidence() -> None:
    stale_mean: Any = [{"metric": "mean", "value": 49.0, "detail": "mean([49])", "inputs": [49.0]}]

    annotated, _ = flag_unverified_numbers("Mean $49.", [_OLD, _NEW], stale_mean)

    assert OUTDATED_MARK in annotated


def test_citing_a_source_date_is_not_an_unverified_figure() -> None:
    annotated, events = flag_unverified_numbers("As of 2026-08 Acme charges $55.", [_OLD, _NEW], [])

    assert annotated == "As of 2026-08 Acme charges $55."
    assert events == []


def test_outdated_figures_are_reported_with_their_sentence() -> None:
    found = figures_only_outdated_sources_support(
        "Acme now charges $55. Previously it was $49.", [_OLD, _NEW]
    )

    assert found == [("$49", "Previously it was $49.")]


def test_a_source_year_does_not_ground_figures_near_it() -> None:
    annotated, events = flag_unverified_numbers("Acme has 2,040 customers.", [_OLD, _NEW], [])

    assert UNVERIFIED_MARK.strip() in annotated
    assert [e["rule"] for e in events] == ["unverified_numbers"]


def test_a_source_year_does_not_ground_tool_inputs() -> None:
    assert ungrounded_inputs([2040.0, 2026.0], [_OLD, _NEW]) == [2040.0, 2026.0]


def test_an_old_figure_the_report_labels_as_outdated_is_left_alone() -> None:
    text = "Acme charges $55. The earlier price was $49 (as of 2026-03)."

    annotated, events = flag_unverified_numbers(text, [_OLD, _NEW], [])

    assert annotated == text
    assert events == []


@pytest.mark.parametrize(
    ("sentence", "labelled"),
    [
        ("Acme raised the Starter price from $49 to $55.", True),
        ("Now $55, up from $49.", True),
        ("The price increased from $49.", True),
        ("Price range (historical): $49 to $55.", True),
        ("Starter was $49 in March 2026.", True),
        ("Starter was $49 (as of 2026-03).", True),
        ("Minimum price per seat: $49.", False),
        ("Acme charges $49 in 2026.", False),  # a current source shares that year
    ],
)
def test_ways_a_report_can_label_an_old_figure(sentence: str, labelled: bool) -> None:
    assert is_labelled_outdated(sentence, [_OLD, _NEW]) is labelled


def test_the_old_year_alone_labels_a_figure_when_no_current_source_shares_it() -> None:
    old: Any = {**_OLD, "as_of": "2024-11"}

    assert is_labelled_outdated("In 2024 it was $49.", [old, _NEW])
