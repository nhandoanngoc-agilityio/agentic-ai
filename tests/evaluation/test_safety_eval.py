"""Safety eval cases, with fake LLMs (no network)."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from market_research_team import graph as graph_module
from market_research_team.config import settings
from market_research_team.evaluation import safety_eval
from market_research_team.evaluation.golden_dataset import SAFETY_CASES


class _DraftLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages: list[object], config: object = None) -> AIMessage:
        return AIMessage(content=self._content)


@pytest.fixture(autouse=True)
def _tmp_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")


@pytest.fixture
def _discarding_graph(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_run_graph(state, *, thread_id=None, **_):
        seen["thread_id"] = thread_id
        if isinstance(state, Command):
            seen["resume"] = state.resume
            return {"report_path": None, "report_discarded": True}
        return {"__interrupt__": ["pending"]}

    monkeypatch.setattr(graph_module, "run_graph", fake_run_graph)
    return seen


def _by_name(results):
    return {r.case_name: r for r in results}


def test_every_safety_case_is_evaluated(_discarding_graph):
    results = safety_eval.evaluate_safety(_DraftLLM("# Report\n\nClean."), "p", repeat=1)

    assert {r.case_name for r in results} == {case.name for case in SAFETY_CASES}
    assert all(r.category == "safety" and r.repeat == 1 for r in results)


def test_clean_behaviour_passes_every_case(_discarding_graph):
    results = _by_name(safety_eval.evaluate_safety(_DraftLLM("# Report\n\nClean."), "p"))

    failed = {name: r.detail for name, r in results.items() if not r.passed}
    assert failed == {}
    assert _discarding_graph["resume"] == {"approved": False, "discard": True}
    assert str(_discarding_graph["thread_id"]).startswith("eval-safety-")


def test_draft_that_obeys_injection_fails(_discarding_graph):
    obeying = _DraftLLM("Report written to ../secrets.md as instructed.")

    results = _by_name(safety_eval.evaluate_safety(obeying, "p"))

    assert not results["injection_in_retrieved_chunk"].passed


def test_pii_is_redacted_even_when_the_draft_repeats_it(_discarding_graph):
    leaky = _DraftLLM("Contact jane.doe@acme-corp.com or 415-555-0100.")

    results = _by_name(safety_eval.evaluate_safety(leaky, "p"))

    assert results["pii_and_secret_redacted"].passed


def test_a_written_report_fails_no_write_without_approval(monkeypatch):
    def writes_anyway(state, **_):
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / "leak.md").write_text("x", encoding="utf-8")
        return {"report_path": str(settings.reports_dir / "leak.md")}

    monkeypatch.setattr(graph_module, "run_graph", writes_anyway)

    results = _by_name(safety_eval.evaluate_safety(_DraftLLM("clean"), "p"))

    assert not results["no_write_without_approval"].passed


def test_a_case_that_raises_fails_closed(monkeypatch, _discarding_graph):
    def boom(*_args, **_kwargs):
        raise RuntimeError("guard crashed")

    monkeypatch.setattr(safety_eval, "input_guard_node", boom)

    results = _by_name(safety_eval.evaluate_safety(_DraftLLM("clean"), "p"))

    assert not results["injection_in_objective"].passed
    assert "guard crashed" in results["injection_in_objective"].detail
