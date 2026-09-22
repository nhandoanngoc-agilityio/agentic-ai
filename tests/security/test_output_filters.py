from market_research_team.security.output_filters import (
    UNVERIFIED_MARK,
    apply_output_guardrails,
    flag_unverified_numbers,
    redact_pii,
    scrub_secrets,
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
