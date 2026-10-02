"""The candidate review queue under data/regression_candidates/."""

import json
from pathlib import Path

import pytest

from market_research_team.feedback.candidates import (
    CandidateQueue,
    PromotionError,
    candidate_id,
    scrub_text,
)
from market_research_team.feedback.regressions import Expectations, load_regressions
from market_research_team.feedback.signals import RunEvidence

NOW = "2026-09-30T10:00:00+00:00"


def _ev(thread="t1", objective="Compare  Acme pricing", **kw) -> RunEvidence:
    return RunEvidence(
        thread, objective=objective, agent_version="0.1.0+a", sources=["audit"], **kw
    )


def test_candidate_id_normalises_case_and_whitespace():
    assert candidate_id("Compare  Acme pricing") == candidate_id("compare acme PRICING ")
    assert len(candidate_id("x")) == 12


def test_scrub_text_removes_pii_and_secrets():
    text = "mail jane.doe@acme-corp.com key sk-ant-abcdefghijklmnopqrstuvwxyz"
    cleaned = scrub_text(text)
    assert "jane.doe@acme-corp.com" not in cleaned and "sk-ant-" not in cleaned
    assert "[redacted]" in cleaned


def test_add_writes_a_scrubbed_pending_candidate(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    ev = _ev(
        objective="Email jane.doe@acme-corp.com the Acme pricing",
        decisions=[{"approved": False, "feedback": "call 415-555-0100"}],
    )

    assert queue.add(ev, ["rejected"]) == "new"

    (candidate,), invalid = queue.list_pending()
    assert invalid == []
    assert "jane.doe" not in candidate.objective
    assert candidate.triggers == ["rejected"]
    assert candidate.occurrences == 1
    assert "415-555-0100" not in json.dumps(candidate.evidence)
    assert candidate.expectations == Expectations()


def test_add_same_objective_merges_and_same_thread_is_not_duplicated(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev("t1"), ["thumbs_down"])

    assert queue.add(_ev("t2", objective="compare acme pricing"), ["rejected"]) == "merged"
    assert queue.add(_ev("t2", objective="compare acme pricing"), ["rejected"]) == "merged"

    (candidate,), _ = queue.list_pending()
    assert candidate.occurrences == 2
    assert candidate.triggers == ["thumbs_down", "rejected"]
    assert [e["thread_id"] for e in candidate.evidence] == ["t1", "t2"]


def test_promoted_and_rejected_ids_are_not_proposed_again(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev("t1", objective="first"), ["thumbs_down"])
    queue.add(_ev("t2", objective="second"), ["thumbs_down"])
    first, second = candidate_id("first"), candidate_id("second")
    c = queue.load(first)
    c.expectations = Expectations(required_facts=["$49"])
    queue.save(c)
    queue.promote(first, tmp_path / "regressions.jsonl", now=NOW)
    queue.reject(second, "not reproducible")

    assert queue.add(_ev("t3", objective="first"), ["thumbs_down"]) == "skipped"
    assert queue.add(_ev("t4", objective="second"), ["thumbs_down"]) == "skipped"
    assert queue.list_pending() == ([], [])


def test_promote_appends_regression_with_origin_and_moves_file(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev(), ["thumbs_down"])
    cid = candidate_id("Compare Acme pricing")
    c = queue.load(cid)
    c.expectations = Expectations(forbidden_substrings=["$59"])
    c.note = "invented price"
    queue.save(c)

    entry = queue.promote(cid, tmp_path / "evals" / "regressions.jsonl", now=NOW)

    assert entry.origin == {
        "triggers": ["thumbs_down"],
        "harvested_at": NOW,
        "agent_versions": ["0.1.0+a"],
        "note": "invented price",
    }
    assert load_regressions(tmp_path / "evals" / "regressions.jsonl") == [entry]
    assert (tmp_path / "promoted" / f"{cid}.json").exists()
    assert not (tmp_path / "pending" / f"{cid}.json").exists()


@pytest.mark.parametrize(
    ("expectations", "message"),
    [
        (Expectations(), "at least one expectation"),
        (Expectations(must_block=True, required_facts=["x"]), "must_block"),
        (Expectations(required_facts=["jane@acme.com"]), "personal data"),
    ],
)
def test_promote_refusals(tmp_path: Path, expectations, message):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev(objective="Compare Acme pricing"), ["thumbs_down"])
    cid = candidate_id("Compare Acme pricing")
    c = queue.load(cid)
    c.expectations = expectations
    queue.save(c)

    with pytest.raises(PromotionError, match=message):
        queue.promote(cid, tmp_path / "regressions.jsonl", now=NOW)
    assert not (tmp_path / "regressions.jsonl").exists()


def test_promote_refuses_an_id_already_in_the_file(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev(), ["thumbs_down"])
    cid = candidate_id("Compare Acme pricing")
    c = queue.load(cid)
    c.expectations = Expectations(min_findings=1)
    queue.save(c)
    regressions = tmp_path / "regressions.jsonl"
    regressions.write_text(
        json.dumps({"id": cid, "objective": "o", "expectations": {}, "origin": {}}) + "\n"
    )

    with pytest.raises(PromotionError, match="already"):
        queue.promote(cid, regressions, now=NOW)


def test_invalid_candidate_files_are_listed_not_loaded(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    (tmp_path / "pending").mkdir(parents=True)
    (tmp_path / "pending" / "broken.json").write_text("{nope")

    assert queue.list_pending() == ([], ["broken"])


def test_last_harvest_marker(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    assert queue.last_harvest() is None
    queue.set_last_harvest(NOW)
    assert queue.last_harvest() == NOW


def test_candidate_id_ignores_personal_data():
    assert candidate_id("Email jane@acme.com about pricing") == candidate_id(
        "Email bob@corp.io about pricing"
    )


def test_promote_refuses_personal_data_in_the_note(tmp_path: Path):
    queue = CandidateQueue(tmp_path)
    queue.add(_ev(objective="Compare Acme pricing"), ["thumbs_down"])
    cid = candidate_id("Compare Acme pricing")
    c = queue.load(cid)
    c.expectations = Expectations(min_findings=1)
    c.note = "user jane@acme.com complained"
    queue.save(c)

    with pytest.raises(PromotionError, match="personal data"):
        queue.promote(cid, tmp_path / "regressions.jsonl", now=NOW)
