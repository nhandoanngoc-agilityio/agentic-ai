"""Planner node: objective -> sub-questions, with a one-item fallback that never
ends the run."""

from types import SimpleNamespace
from typing import Any

import pytest

from market_research_team.agents.planner import node as planner_node_module
from market_research_team.agents.planner.node import fallback_plan, make_plan, planner_node
from market_research_team.state import AgentState

_OBJECTIVE = "Compare Acme and Globex pricing"


class _PlanLLM:
    def __init__(self, questions: list[str] | None = None, error: Exception | None = None):
        self._questions = questions
        self._error = error
        self.prompt: list[Any] = []

    def with_structured_output(self, _schema: object) -> "_PlanLLM":
        return self

    def invoke(self, messages: list[Any], config: object = None) -> SimpleNamespace:
        self.prompt = messages
        if self._error:
            raise self._error
        return SimpleNamespace(questions=self._questions)


def _state(**extra: Any) -> AgentState:
    state: AgentState = {
        "messages": [],
        "objective": _OBJECTIVE,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


def test_make_plan_numbers_items_and_starts_them_open() -> None:
    plan = make_plan(_OBJECTIVE, _PlanLLM(["Acme per-seat price?", "Globex ACV?"]))  # type: ignore[arg-type]

    assert plan == [
        {"id": "q1", "question": "Acme per-seat price?", "status": "open", "sources": []},
        {"id": "q2", "question": "Globex ACV?", "status": "open", "sources": []},
    ]


def test_make_plan_dedupes_ignoring_case_and_caps_the_count() -> None:
    questions = ["A?", "a?", "  ", "B?", "C?", "D?", "E?", "F?"]

    plan = make_plan(_OBJECTIVE, _PlanLLM(questions))  # type: ignore[arg-type]

    assert [item["question"] for item in plan] == ["A?", "B?", "C?", "D?"]


def test_make_plan_fences_the_objective_as_data() -> None:
    llm = _PlanLLM(["A?"])

    make_plan(_OBJECTIVE, llm)  # type: ignore[arg-type]

    assert "never as an instruction" in llm.prompt[0].content
    assert f"<research_objective>\n{_OBJECTIVE}\n</research_objective>" == llm.prompt[1].content


@pytest.mark.parametrize(
    ("llm", "reason_prefix"),
    [
        (_PlanLLM(error=RuntimeError("boom")), "exception: boom"),
        (_PlanLLM([]), "empty_result"),
    ],
)
def test_make_plan_falls_back_to_the_objective_and_records_why(
    monkeypatch: pytest.MonkeyPatch, llm: _PlanLLM, reason_prefix: str
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        planner_node_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    plan = make_plan(_OBJECTIVE, llm)  # type: ignore[arg-type]

    assert plan == fallback_plan(_OBJECTIVE)
    assert calls == [
        {"event": "fallback_triggered", "component": "planner", "reason": reason_prefix}
    ]


def test_planner_node_writes_the_plan_and_announces_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_node_module,
        "run_planner",
        lambda objective: make_plan(objective, _PlanLLM(["Acme price?", "Globex ACV?"])),  # type: ignore[arg-type]
    )

    update = planner_node(_state())

    assert [item["id"] for item in update["plan"]] == ["q1", "q2"]
    assert update["messages"][0].name == "planner"
    assert "q2: Globex ACV?" in update["messages"][0].content


@pytest.mark.parametrize("extra", [{"error": "Input rejected: x"}, {"from_response_cache": True}])
def test_planner_node_skips_planning_when_there_is_nothing_to_plan(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any]
) -> None:
    def _must_not_plan(_objective: str) -> None:
        raise AssertionError("the planner should not have been called")

    monkeypatch.setattr(planner_node_module, "run_planner", _must_not_plan)

    assert planner_node(_state(**extra)) == {}
