"""validate_insights: what Analytics may hand Reporting."""

from typing import Any

from market_research_team.config import settings
from market_research_team.security.output_filters import INSIGHT_MAX_CHARS, validate_insights

_FINDINGS: Any = [
    {"source": "acme.md", "content": "Acme Starter is $55 per seat.", "relevance_score": 1.0},
    {"source": "initech.md", "content": "Initech Team is $15 per seat.", "relevance_score": 1.0},
]
_RESULTS: Any = [
    {
        "metric": "percent_change",
        "value": 266.67,
        "detail": "d",
        "inputs": [15.0, 55.0],
        "id": "r1",
    },
    {"metric": "mean", "value": 35.0, "detail": "d", "inputs": [15.0, 55.0], "id": "r2"},
]


def _reasons(events: list[Any]) -> list[str]:
    return [event["detail"].split(":")[0] for event in events]


def test_a_grounded_insight_is_kept_with_its_valid_ids() -> None:
    text = "Acme costs 266.67% more per seat than Initech ($55 vs $15)."

    insights, events = validate_insights(
        [{"text": text, "result_ids": ["r1", "r9"]}], _RESULTS, _FINDINGS
    )

    assert insights == [{"text": text, "result_ids": ["r1"]}]
    assert events == []


def test_an_insight_without_figures_is_kept() -> None:
    insights, _ = validate_insights(
        [{"text": "Acme is the most expensive vendor.", "result_ids": ["r2"]}], _RESULTS, _FINDINGS
    )

    assert len(insights) == 1


def test_a_rounded_percentage_is_grounded() -> None:
    insights, events = validate_insights(
        [{"text": "Acme is about 266.7% pricier.", "result_ids": ["r1"]}], _RESULTS, _FINDINGS
    )

    assert insights and not events


def test_an_insight_citing_no_known_result_is_dropped() -> None:
    insights, events = validate_insights(
        [{"text": "Acme leads.", "result_ids": ["r7"]}], _RESULTS, _FINDINGS
    )

    assert insights == []
    assert _reasons(events) == ["no valid result id"]
    assert events[0]["rule"] == "ungrounded_insight" and events[0]["layer"] == "tool"


def test_an_over_long_insight_is_dropped_not_cut() -> None:
    text = "Acme " + "x" * INSIGHT_MAX_CHARS

    insights, events = validate_insights(
        [{"text": text, "result_ids": ["r1"]}], _RESULTS, _FINDINGS
    )

    assert insights == [] and _reasons(events) == ["too long"]


def test_an_insight_with_an_ungrounded_figure_is_dropped() -> None:
    insights, events = validate_insights(
        [{"text": "Acme has 2,040 customers.", "result_ids": ["r1"]}], _RESULTS, _FINDINGS
    )

    assert insights == []
    assert events[0]["detail"].startswith("ungrounded figure 2,040")


def test_a_result_from_an_outdated_input_does_not_ground_an_insight() -> None:
    old: Any = {
        "source": "acme.md",
        "content": "Starter $49.",
        "relevance_score": 1.0,
        "entity": "Acme",
        "topic": "pricing",
        "as_of": "2026-03",
    }
    new: Any = {**old, "source": "bench.md", "content": "Starter $55.", "as_of": "2026-08"}
    results: Any = [
        {"metric": "mean", "value": 52.0, "detail": "d", "inputs": [49.0, 55.0], "id": "r1"}
    ]

    insights, events = validate_insights(
        [{"text": "Acme averages $52 per seat.", "result_ids": ["r1"]}], results, [old, new]
    )

    assert insights == [] and events[0]["detail"].startswith("ungrounded figure")


def test_insights_beyond_the_limit_are_dropped() -> None:
    raw = [{"text": f"Point {n}.", "result_ids": ["r1"]} for n in range(7)]

    insights, events = validate_insights(raw, _RESULTS, _FINDINGS)

    assert len(insights) == settings.run_policy.max_insights == 5
    assert _reasons(events) == ["over the limit", "over the limit"]


def test_malformed_input_never_raises() -> None:
    assert validate_insights(None, _RESULTS, _FINDINGS) == ([], [])
    assert validate_insights([], _RESULTS, _FINDINGS) == ([], [])

    insights, events = validate_insights("a string", _RESULTS, _FINDINGS)
    assert insights == [] and _reasons(events) == ["malformed submission"]

    insights, events = validate_insights([{"text": 3}, "x", {"text": "ok"}], _RESULTS, _FINDINGS)
    assert insights == [] and _reasons(events) == ["malformed insight"] * 3


def test_a_percentage_rounded_to_a_whole_number_is_grounded() -> None:
    results: Any = [
        {"metric": "percent_change", "value": 12.24, "detail": "d", "inputs": [], "id": "r1"}
    ]

    insights, events = validate_insights(
        [{"text": "Prices grew 12% year on year.", "result_ids": ["r1"]}], results, _FINDINGS
    )

    assert insights and not events


def test_a_submission_sent_as_a_json_string_is_parsed() -> None:
    raw = '[{"text": "Acme is the most expensive vendor.", "result_ids": ["r2"]}]'

    insights, events = validate_insights(raw, _RESULTS, _FINDINGS)

    assert len(insights) == 1 and not events


_GLOBEX: Any = [
    {"source": "globex.md", "content": "ACV between $150K and $400K.", "relevance_score": 1.0}
]
_GLOBEX_RESULTS: Any = [
    {"metric": "minimum", "value": 150000.0, "detail": "d", "inputs": [150000.0], "id": "r1"},
    {"metric": "maximum", "value": 400000.0, "detail": "d", "inputs": [400000.0], "id": "r2"},
]


def test_ids_written_into_the_text_are_recovered_and_removed_from_it() -> None:
    raw = [
        {
            "text": "Globex's minimum annual contract value is $150,000 (r1) and maximum "
            "annual contract value is $400,000 (r2)."
        }
    ]

    insights, events = validate_insights(raw, _GLOBEX_RESULTS, _GLOBEX)

    assert insights == [
        {
            "text": "Globex's minimum annual contract value is $150,000 and maximum "
            "annual contract value is $400,000.",
            "result_ids": ["r1", "r2"],
        }
    ]
    assert events == []


def test_an_empty_id_list_also_falls_back_to_ids_in_the_text() -> None:
    raw = [{"text": "Globex deals start at $150K (from r1).", "result_ids": []}]

    insights, _ = validate_insights(raw, _GLOBEX_RESULTS, _GLOBEX)

    assert insights == [{"text": "Globex deals start at $150K.", "result_ids": ["r1"]}]


def test_a_text_citing_no_id_without_result_ids_is_still_dropped() -> None:
    insights, events = validate_insights(
        [{"text": "Globex deals start at $150K."}], _GLOBEX_RESULTS, _GLOBEX
    )

    assert insights == [] and _reasons(events) == ["malformed insight"]
