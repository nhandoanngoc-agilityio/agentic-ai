"""Topics, entities, supersession and labels for dated findings."""

from typing import Any

import pytest

from market_research_team.retrieval.evidence import (
    finding_key,
    finding_label,
    headings_of,
    resolve_entity,
    superseded,
    topic_for,
)


def _f(content: str, source: str, **fields: Any) -> Any:
    return {"source": source, "content": content, "relevance_score": 1.0, **fields}


@pytest.mark.parametrize(
    ("headings", "topic"),
    [
        (["Acme Analytics — Competitor Profile", "Product and Pricing"], "pricing"),
        (["X", "Leadership and Headcount"], "headcount"),
        (["Business Intelligence Market — Overview", "Market Size and Growth"], "market_size"),
        (["X", "Strengths"], "strengths"),
        (["X", "Weaknesses"], "weaknesses"),
        (["X", "Recent Moves"], "recent_moves"),
        (["X", "Customer Ratings", "Acme"], "customers"),
        (["BI Pricing Benchmark", "Per-Seat Pricing by Vendor", "Acme"], "pricing"),
        (["X", "Company Overview"], "other"),
        (["X", "Planning Horizon"], "other"),  # "plan" only as a whole word
        ([], "other"),
    ],
)
def test_topic_comes_from_the_deepest_matching_heading(headings: list[str], topic: str) -> None:
    assert topic_for(headings) == topic


def test_headings_of_reads_h1_to_h3_in_order() -> None:
    assert headings_of({"h1": "A", "h3": "C", "source": "x"}) == ["A", "C"]


@pytest.mark.parametrize(
    ("doc_entity", "headings", "entity"),
    [
        ("Acme", ["Acme", "Pricing"], "Acme"),
        ("Multiple", ["Benchmark", "Pricing", "Acme"], "Acme"),
        ("Multiple", ["Benchmark", "Pricing", "ACME"], "Acme"),  # case differs from profile
        ("Multiple", ["Benchmark", "Methodology"], "Market"),
        (None, ["Acme"], None),
    ],
)
def test_entity_resolution(doc_entity: str | None, headings: list[str], entity: str | None) -> None:
    assert resolve_entity(doc_entity, headings, {"Acme", "Globex"}) == entity


_ACME_OLD = _f("Starter $49 per seat.", "acme.md", entity="Acme", topic="pricing", as_of="2026-03")
_ACME_NEW = _f("Starter $55 per seat.", "bench.md", entity="Acme", topic="pricing", as_of="2026-08")


def test_a_newer_finding_on_the_same_entity_and_topic_supersedes() -> None:
    assert superseded([_ACME_OLD, _ACME_NEW]) == {finding_key(_ACME_OLD): "bench.md"}


@pytest.mark.parametrize(
    "other",
    [
        _f("Starter $55.", "b.md", entity="Globex", topic="pricing", as_of="2026-08"),
        _f("Starter $55.", "b.md", entity="Acme", topic="headcount", as_of="2026-08"),
        _f("Starter $55.", "b.md", entity="Acme", topic="other", as_of="2026-08"),
        _f("Starter $55.", "b.md", entity="Acme", topic="pricing"),  # undated
        _f("Starter $55.", "b.md", entity="Acme", topic="pricing", as_of="2026-03"),  # same date
        _f("Numbers in the paid edition.", "t.md", entity="Acme", topic="pricing", as_of="2026-09"),
    ],
)
def test_nothing_is_superseded_without_a_newer_figure_on_the_same_scope(other: Any) -> None:
    assert superseded([_ACME_OLD, other]) == {}


def test_labels_show_provenance_and_render_undated_findings_as_before() -> None:
    by = superseded([_ACME_OLD, _ACME_NEW])

    assert finding_label(_ACME_NEW, by) == "bench.md · Acme · pricing · as of 2026-08"
    assert finding_label(_ACME_OLD, by) == (
        "acme.md · Acme · pricing · as of 2026-03 (superseded by bench.md)"
    )
    assert finding_label(_f("x", "plain.md"), {}) == "plain.md"
