import pytest

from market_research_team.evaluation import checks


def test_check_query_count_within_range() -> None:
    passed, _ = checks.check_query_count(["a", "b"], min_count=1, max_count=4)
    assert passed


def test_check_query_count_outside_range() -> None:
    passed, _ = checks.check_query_count([], min_count=1, max_count=4)
    assert not passed


def test_check_keyword_coverage_matches_case_insensitively() -> None:
    passed, detail = checks.check_keyword_coverage(["Acme Pricing Overview"], ["pricing"])
    assert passed
    assert "pricing" in detail.lower()


def test_check_keyword_coverage_fails_without_match() -> None:
    passed, _ = checks.check_keyword_coverage(["unrelated text"], ["pricing", "acme"])
    assert not passed


def test_check_keyword_coverage_passes_trivially_with_no_requirement() -> None:
    passed, _ = checks.check_keyword_coverage(["anything"], [])
    assert passed


def test_check_contains_all_reports_missing_items() -> None:
    passed, detail = checks.check_contains_all(
        "Objective\nFindings", ["Objective", "Recommendation"]
    )
    assert not passed
    assert "Recommendation" in detail


def test_check_contains_all_passes_when_everything_present() -> None:
    passed, _ = checks.check_contains_all("Objective and Findings here", ["Objective", "Findings"])
    assert passed


def test_check_min_length() -> None:
    passed, _ = checks.check_min_length([1, 2, 3], 2, "items")
    assert passed

    passed, _ = checks.check_min_length([], 1, "items")
    assert not passed


def test_check_source_coverage_passes_when_expected_source_retrieved() -> None:
    findings = [{"source": "competitor_acme.md", "content": "x", "relevance_score": 0.9}]
    passed, _ = checks.check_source_coverage(findings, ["competitor_acme.md"], min_hits=1)
    assert passed


def test_check_source_coverage_fails_when_expected_source_missing() -> None:
    findings = [{"source": "market_overview.md", "content": "x", "relevance_score": 0.9}]
    passed, _ = checks.check_source_coverage(findings, ["competitor_acme.md"], min_hits=1)
    assert not passed


def test_check_source_coverage_passes_trivially_with_no_requirement() -> None:
    passed, _ = checks.check_source_coverage([], [], min_hits=1)
    assert passed


def test_check_source_coverage_requires_min_hits() -> None:
    findings = [{"source": "competitor_acme.md", "content": "x", "relevance_score": 0.9}]
    passed, _ = checks.check_source_coverage(
        findings, ["competitor_acme.md", "competitor_globex.md"], min_hits=2
    )
    assert not passed


def test_check_any_value_matches_within_tolerance() -> None:
    results = [{"metric": "range", "value": 250000.0, "detail": "d"}]
    passed, _ = checks.check_any_value_matches(results, [250000.0], tolerance=1.0)
    assert passed


def test_check_any_value_matches_fails_when_no_result_is_close() -> None:
    results = [{"metric": "made_up", "value": 999.0, "detail": "d"}]
    passed, _ = checks.check_any_value_matches(results, [250000.0], tolerance=1.0)
    assert not passed


def test_check_any_value_matches_fails_on_empty_results() -> None:
    passed, _ = checks.check_any_value_matches([], [250000.0], tolerance=1.0)
    assert not passed


def test_check_any_value_matches_passes_trivially_with_no_expectation() -> None:
    passed, _ = checks.check_any_value_matches([], [], tolerance=1.0)
    assert passed


# --- agentic-behaviour checks ----------------------------------------------------


def test_check_supervisor_route_requires_the_expected_decision_and_a_focused_gap() -> None:
    from market_research_team.evaluation.checks import check_supervisor_route

    assert check_supervisor_route("research", "Globex ACV?", ("research",), ["globex"])[0]
    assert not check_supervisor_route("analytics", None, ("research",), ["globex"])[0]
    passed, detail = check_supervisor_route("research", "Acme seats?", ("research",), ["globex"])
    assert not passed and "focus missing ['globex']" in detail
    # A focus is only checked on a hand-back to Research.
    assert check_supervisor_route("reporting", None, ("reporting",), ["globex"])[0]
    # The right step reached through a fallback is not the model's decision.
    assert not check_supervisor_route("reporting", None, ("reporting",), [], "fallback")[0]


def test_check_plan_size_coverage_and_no_restated_comparison() -> None:
    from market_research_team.evaluation.checks import check_plan

    good = ["What does Acme charge per seat?", "What is Globex's contract value?"]
    assert check_plan(good, min_items=2, max_items=5, must_mention=["acme", "globex"])[0]

    passed, detail = check_plan(
        ["How do Acme and Globex prices compare?", "Acme customers?"],
        min_items=2,
        max_items=5,
        must_mention=["acme", "globex"],
    )
    assert not passed and "restated comparisons: ['How do Acme" in detail
    assert not check_plan(["Acme price?"], min_items=2, max_items=5, must_mention=[])[0]
    # Naming both companies as context is not a comparison.
    market = "What is the size of the market in which Acme and Globex operate?"
    assert check_plan([*good, market], min_items=2, max_items=5, must_mention=["acme", "globex"])[0]
    assert not check_plan(good[:1] * 2, min_items=2, max_items=5, must_mention=["globex"])[0]


def test_check_inputs_grounded_names_the_invented_inputs() -> None:
    from market_research_team.evaluation.checks import check_inputs_grounded

    findings = [
        {"source": "g.md", "content": "ACV between $150K and $400K.", "relevance_score": 1.0}
    ]
    grounded = [{"metric": "mean", "value": 275000.0, "detail": "", "inputs": [150000.0, 400000.0]}]
    invented = [{"metric": "mean", "value": 150.0, "detail": "", "inputs": [120.0, 180.0]}]

    assert check_inputs_grounded(grounded, findings)[0]  # type: ignore[arg-type]
    passed, detail = check_inputs_grounded(invented, findings)  # type: ignore[arg-type]
    assert not passed and "'mean': [120.0, 180.0]" in detail


def test_check_markdown_table_needs_a_separator_row() -> None:
    from market_research_team.evaluation.checks import check_markdown_table

    assert check_markdown_table("| Vendor | Price |\n|---|---|\n| Acme | $49 |")[0]
    assert check_markdown_table("Vendor | Price\n:--- | ---:\nAcme | $49")[0]
    assert not check_markdown_table("Acme | $49 per seat")[0]


_PLAN = [
    {"id": "q1", "question": "Acme price?", "status": "answered", "sources": ["a.md"]},
    {"id": "q2", "question": "Globex ACV?", "status": "unanswerable", "sources": []},
]
_GOOD_ROUTE = [
    {"next": "research", "decided_by": "rule"},
    {"next": "research", "decided_by": "llm", "focus": "Globex ACV?"},
    {"next": "analytics", "decided_by": "rule"},
    {"next": "reporting", "decided_by": "llm"},
    {"next": "FINISH", "decided_by": "rule"},
]


def test_check_trajectory_passes_a_clean_run() -> None:
    from market_research_team.evaluation.checks import check_trajectory

    passed, detail = check_trajectory(_PLAN, _GOOD_ROUTE, "## Open questions\n- Globex ACV?")  # type: ignore[arg-type]

    assert passed, detail
    assert detail.startswith("route: research(rule) -> research(llm) -> analytics(rule)")


@pytest.mark.parametrize(
    ("route", "report", "problem"),
    [
        (_GOOD_ROUTE, "no open questions here", "unanswered but not in the report"),
        (
            [*_GOOD_ROUTE[:3], {"next": "reporting", "decided_by": "fallback"}],
            "Globex ACV?",
            "1 fallback route(s)",
        ),
        (
            [{"next": "research", "decided_by": "llm"}],
            "Globex ACV?",
            "1 hand-back(s) without a focus",
        ),
        (
            [{"next": "research", "decided_by": "rule"}] * 4,
            "Globex ACV?",
            "4 research passes for 2 plan item(s)",
        ),
        (
            [
                *_GOOD_ROUTE[:2],
                {
                    "next": "analytics",
                    "decided_by": "rule",
                    "rationale": "visit cap reached: moving on to analysis",
                },
            ],
            "Globex ACV?",
            "the visit cap forced progress",
        ),
        (
            [
                {"next": "research", "decided_by": "llm", "focus": "Globex ACV?"},
                {"next": "research", "decided_by": "llm", "focus": "Globex ACV?"},
            ],
            "Globex ACV?",
            "targeted more than once: ['Globex ACV?']",
        ),
    ],
)
def test_check_trajectory_names_each_problem(route, report, problem) -> None:
    from market_research_team.evaluation.checks import check_trajectory

    passed, detail = check_trajectory(_PLAN, route, report)  # type: ignore[arg-type]

    assert not passed
    assert problem in detail


def test_check_trajectory_fails_without_a_plan() -> None:
    from market_research_team.evaluation.checks import check_trajectory

    assert "no plan" in check_trajectory([], _GOOD_ROUTE, "")[1]


_OLD = {
    "source": "acme.md",
    "content": "Starter $49.",
    "relevance_score": 1.0,
    "entity": "Acme",
    "topic": "pricing",
    "as_of": "2026-03",
}
_NEW = {**_OLD, "source": "bench.md", "content": "Starter $55.", "as_of": "2026-08"}


@pytest.mark.parametrize(
    ("report", "passed"),
    [
        ("Acme charges $55 per seat.", True),
        ("Acme charges $55. The outdated price was $49.", True),
        ("Acme charges $55; in 2026-03 it was $49.", True),
        ("Acme charges $49 per seat.", False),
    ],
)
def test_check_freshness_requires_older_figures_to_be_labelled(report: str, passed: bool) -> None:
    ok, detail = checks.check_freshness(report, [_OLD, _NEW])  # type: ignore[list-item]

    assert ok is passed, detail


def test_check_freshness_is_a_no_op_without_conflicts() -> None:
    assert checks.check_freshness("Anything $49.", [_OLD]) == (True, "no conflicting sources")  # type: ignore[list-item]


def test_check_inputs_current_flags_outdated_inputs() -> None:
    stale = [{"metric": "mean", "value": 49.0, "detail": "mean([49])", "inputs": [49.0]}]
    fresh = [{"metric": "mean", "value": 55.0, "detail": "mean([55])", "inputs": [55.0]}]

    assert not checks.check_inputs_current(stale, [_OLD, _NEW])[0]  # type: ignore[arg-type]
    assert checks.check_inputs_current(fresh, [_OLD, _NEW])[0]  # type: ignore[arg-type]
