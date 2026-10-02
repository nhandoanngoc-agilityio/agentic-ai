"""Which production runs count as failures worth a regression case."""

import pytest

from market_research_team.feedback.signals import RunEvidence
from market_research_team.feedback.triggers import detect_triggers, suggest_expectations


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (RunEvidence("t", ratings=["up"]), []),
        (RunEvidence("t", ratings=["down"]), ["thumbs_down"]),
        (RunEvidence("t", decisions=[{"approved": False, "feedback": "x"}]), ["rejected"]),
        (
            RunEvidence("t", decisions=[{"approved": False}, {"approved": False}]),
            ["rejected", "repeated_rejection"],
        ),
        (RunEvidence("t", decisions=[{"discard": True}, {"approved": True}]), []),
        (RunEvidence("t", error="Recursion limit reached: 20"), ["run_error"]),
        (
            RunEvidence(
                "t",
                error="Input rejected: matched injection pattern",
                guardrail_events=[{"layer": "input", "rule": "validate_objective"}],
            ),
            ["blocked_input"],
        ),
        (
            RunEvidence(
                "t", guardrail_events=[{"layer": "retrieval", "rule": "injection_in_chunk"}]
            ),
            ["dropped_chunk"],
        ),
        (RunEvidence("t", fallbacks=["supervisor_router"]), ["fallback"]),
        (RunEvidence("t", langfuse_errors=["boom"]), ["langfuse_error"]),
        (RunEvidence("t", guardrail_events=[{"layer": "output", "rule": "pii_redacted"}]), []),
    ],
)
def test_detect_triggers(evidence, expected):
    assert detect_triggers(evidence) == expected


def test_suggest_must_block_for_blocked_input():
    assert suggest_expectations(["blocked_input"]) == {"must_block": True}
    assert suggest_expectations(["thumbs_down"]) == {}
