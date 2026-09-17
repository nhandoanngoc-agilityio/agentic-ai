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
