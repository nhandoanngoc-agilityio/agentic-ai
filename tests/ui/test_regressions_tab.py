"""Regressions tab handlers (pure functions over a temp queue)."""

from pathlib import Path

import pytest
from gradio_app import regressions_tab as tab

from market_research_team.feedback.candidates import CandidateQueue, candidate_id
from market_research_team.feedback.regressions import load_regressions
from market_research_team.feedback.signals import RunEvidence

CID = candidate_id("Compare Acme pricing")


@pytest.fixture
def queue(tmp_path: Path) -> CandidateQueue:
    q = CandidateQueue(tmp_path / "q")
    q.add(
        RunEvidence(
            "t1",
            objective="Compare Acme pricing",
            agent_version="0.1.0+a",
            decisions=[{"approved": False, "feedback": "price wrong"}],
            guardrail_events=[{"layer": "input", "rule": "validate_objective"}],
            langfuse_trace_ids=["tr1"],
            sources=["audit"],
        ),
        ["rejected", "blocked_input"],
    )
    return q


def test_pending_rows(queue):
    (row,) = tab.pending_rows(queue)
    assert row[0] == CID and "rejected" in row[1] and row[2] == "1"


def test_candidate_details_show_evidence_and_prefill_suggestions(queue):
    markdown, form = tab.candidate_details(queue, CID)
    assert "price wrong" in markdown and "tr1" in markdown
    assert form["must_block"] is True  # from the blocked_input suggestion
    assert tab.candidate_details(queue, "missing")[0].startswith("No pending candidate")


def test_promote_from_form_writes_the_case(queue, tmp_path):
    path = tmp_path / "regressions.jsonl"

    message = tab.promote_from_form(queue, path, CID, False, "$49\n\n", "$59", "2", True, "n")

    assert message.startswith("Promoted")
    (entry,) = load_regressions(path)
    assert entry.expectations.required_facts == ["$49"]
    assert entry.expectations.forbidden_substrings == ["$59"]
    assert entry.expectations.min_findings == 2 and entry.expectations.requires_approval


def test_promote_errors_are_returned_not_raised(queue, tmp_path):
    path = tmp_path / "r.jsonl"
    message = tab.promote_from_form(queue, path, CID, False, "", "", "", False, "")
    assert "at least one expectation" in message
    bad = tab.promote_from_form(queue, path, CID, False, "", "", "x", False, "")
    assert bad.startswith("min_findings")


def test_reject_requires_a_reason(queue):
    assert "reason" in tab.reject_from_form(queue, CID, "  ")
    assert tab.reject_from_form(queue, CID, "noise").startswith("Rejected")
    assert tab.pending_rows(queue) == []


def test_harvest_now_reports_counts(tmp_path):
    q = CandidateQueue(tmp_path / "q")
    assert "Scanned 0 runs" in tab.harvest_now(q, tmp_path / "none.jsonl")
