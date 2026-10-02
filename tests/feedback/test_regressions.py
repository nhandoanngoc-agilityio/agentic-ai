"""The committed regression case file (evals/regressions.jsonl)."""

from pathlib import Path

import pytest

from market_research_team.config import settings
from market_research_team.feedback.regressions import (
    Expectations,
    RegressionEntry,
    append_regression,
    load_regressions,
)


def test_expectations_round_trip_and_emptiness():
    assert Expectations().is_empty()
    e = Expectations(required_facts=["$49"], requires_approval=True)
    assert not e.is_empty()
    assert Expectations.from_dict({**e.to_dict(), "unknown": 1}) == e


def test_append_then_load(tmp_path: Path):
    path = tmp_path / "evals" / "regressions.jsonl"
    entry = RegressionEntry("abc123", "Compare Acme", Expectations(must_block=True), {"n": 1})

    append_regression(path, entry)
    append_regression(path, RegressionEntry("def456", "o", Expectations(min_findings=2), {}))

    loaded = load_regressions(path)
    assert [e.id for e in loaded] == ["abc123", "def456"]
    assert loaded[0] == entry


def test_missing_file_is_empty(tmp_path: Path):
    assert load_regressions(tmp_path / "none.jsonl") == []


def test_malformed_line_names_the_line(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": "a", "objective": "o", "expectations": {}, "origin": {}}\n{oops\n')

    with pytest.raises(ValueError, match="line 2"):
        load_regressions(path)


def test_repo_regressions_file_exists_and_loads():
    assert settings.regressions_path.name == "regressions.jsonl"
    assert settings.regressions_path.exists()
    assert isinstance(load_regressions(settings.regressions_path), list)
