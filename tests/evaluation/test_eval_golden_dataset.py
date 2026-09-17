from market_research_team.evaluation.golden_dataset import (
    ANALYTICS_CASES,
    FULL_PIPELINE_CASES,
    QUERY_REWRITE_CASES,
    REPORTING_CASES,
    SUPERVISOR_DECISION_CASES,
)

_CASE_LISTS = [
    QUERY_REWRITE_CASES,
    SUPERVISOR_DECISION_CASES,
    ANALYTICS_CASES,
    REPORTING_CASES,
    FULL_PIPELINE_CASES,
]


def test_every_case_list_is_non_empty() -> None:
    for cases in _CASE_LISTS:
        assert len(cases) > 0


def test_case_names_are_unique_within_each_list() -> None:
    for cases in _CASE_LISTS:
        names = [case.name for case in cases]
        assert len(names) == len(set(names)), f"duplicate case names: {names}"
