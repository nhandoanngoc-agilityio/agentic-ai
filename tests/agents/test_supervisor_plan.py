"""Supervisor decisions driven by the plan: validated coverage, targeted
hand-backs, a research cut-off once the plan is resolved, and a reason on
every route."""

from types import SimpleNamespace
from typing import Any

import pytest

from market_research_team.agents.supervisor import router as router_module
from market_research_team.agents.supervisor.router import apply_coverage, decide_route
from market_research_team.state import AgentState, PlanItem

_ACME = {"source": "acme.md", "content": "Acme Starter is $49 per seat.", "relevance_score": 1.0}
_RESULT = {"metric": "mean", "value": 49.0, "detail": "d", "entity": None}


def _plan(*statuses: str) -> list[PlanItem]:
    return [
        {"id": f"q{i}", "question": f"Question {i}?", "status": status, "sources": []}  # type: ignore[typeddict-item]
        for i, status in enumerate(statuses, start=1)
    ]


def _state(plan: list[PlanItem], *, analytics: bool = False) -> AgentState:
    return {
        "messages": [],
        "objective": "Compare Acme and Globex pricing",
        "next": "research",
        "research_findings": [_ACME],  # type: ignore[list-item]
        "analytics_results": [_RESULT] if analytics else [],  # type: ignore[list-item]
        "report_path": None,
        "plan": plan,
    }


def _cover(item_id: str, status: str, *sources: str) -> SimpleNamespace:
    return SimpleNamespace(id=item_id, status=status, sources=list(sources))


class _DecisionLLM:
    """Returns one scripted decision and keeps the prompt it was shown."""

    def __init__(self, **decision: Any) -> None:
        self._decision = SimpleNamespace(
            **{"rationale": "", "coverage": [], "focus_id": None, **decision}
        )
        self.prompt: list[Any] = []

    def with_structured_output(self, _schema: object) -> "_DecisionLLM":
        return self

    def invoke(self, messages: list[Any], config: object = None) -> SimpleNamespace:
        self.prompt = messages
        return self._decision


# --- apply_coverage -------------------------------------------------------------


def test_answered_sticks_only_with_a_cited_source_that_is_among_the_findings() -> None:
    plan = _plan("open", "open", "open")

    updated = apply_coverage(
        plan,
        [
            _cover("q1", "answered", "acme.md", "made-up.md"),
            _cover("q2", "answered"),  # no citation
            _cover("q3", "answered", "made-up.md"),  # citation not among findings
        ],
        {"acme.md"},
    )

    assert updated[0] == {**plan[0], "status": "answered", "sources": ["acme.md"]}
    assert updated[1] == plan[1]
    assert updated[2] == plan[2]


def test_unanswerable_is_never_undone_and_unknown_ids_are_ignored() -> None:
    plan = _plan("unanswerable", "answered")
    plan[1]["sources"] = ["acme.md"]

    updated = apply_coverage(
        plan, [_cover("q1", "answered", "acme.md"), _cover("q9", "open")], {"acme.md"}
    )

    assert updated == plan


def test_a_reopened_item_drops_its_sources() -> None:
    plan = _plan("answered")
    plan[0]["sources"] = ["acme.md"]

    assert apply_coverage(plan, [_cover("q1", "open")], {"acme.md"})[0]["sources"] == []


# --- decide_route with a plan ----------------------------------------------------


def test_hand_back_targets_the_named_open_item_and_carries_the_rationale() -> None:
    llm = _DecisionLLM(
        next="research",
        focus_id="q2",
        rationale="Globex is not covered yet.",
        coverage=[_cover("q1", "answered", "acme.md"), _cover("q2", "open")],
    )

    route = decide_route(_state(_plan("open", "open")), llm)  # type: ignore[arg-type]

    assert route.next == "research"
    assert route.focus_id == "q2"
    assert route.research_focus == "Question 2?"
    assert route.decided_by == "llm"
    assert route.rationale == "Globex is not covered yet."
    assert route.plan is not None and route.plan[0]["status"] == "answered"
    assert (
        "<plan>\n- q1 (to judge) Question 1?\n- q2 (to judge) Question 2?\n</plan>"
        in llm.prompt[1].content
    )


def test_hand_back_with_an_invalid_focus_targets_the_first_open_item() -> None:
    llm = _DecisionLLM(next="research", focus_id="q7")

    route = decide_route(_state(_plan("answered", "open", "open")), llm)  # type: ignore[arg-type]

    assert route.focus_id == "q2"


def test_research_is_not_offered_once_every_item_is_resolved() -> None:
    llm = _DecisionLLM(next="reporting")

    state = _state(_plan("answered", "unanswerable"), analytics=True)
    state["findings_version"] = 1  # new findings since the last analysis

    route = decide_route(state, llm)  # type: ignore[arg-type]

    assert route.next == "reporting"
    assert "Allowed next steps: analytics, reporting" in llm.prompt[1].content


def test_research_chosen_after_resolving_the_whole_plan_is_overridden_by_rule() -> None:
    llm = _DecisionLLM(next="research", coverage=[_cover("q1", "answered", "acme.md")])

    route = decide_route(_state(_plan("open"), analytics=True), llm)  # type: ignore[arg-type]

    assert route.next == "reporting"
    assert route.decided_by == "rule"
    assert route.rationale == "no open plan item is left to research"


def test_code_decided_routes_say_so() -> None:
    state = _state(_plan("open"))
    state["research_findings"] = []

    route = decide_route(state, _DecisionLLM(next="analytics"))  # type: ignore[arg-type]

    assert (route.next, route.decided_by) == ("research", "rule")
    assert route.rationale.startswith("no findings yet")


def test_a_failed_routing_call_is_marked_as_a_fallback() -> None:
    class _Broken:
        def with_structured_output(self, _schema: object) -> None:
            raise RuntimeError("down")

    route = decide_route(_state(_plan("open")), _Broken())  # type: ignore[arg-type]

    assert route.decided_by == "fallback"
    assert "down" in route.rationale


# --- supervisor_node -------------------------------------------------------------


def test_supervisor_node_records_the_decision_and_updates_plan_and_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        router_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )
    plan = _plan("answered", "open")
    monkeypatch.setattr(
        router_module,
        "run_supervisor_decision",
        lambda state: router_module.SupervisorRoute(
            "research", "Question 2?", "Globex missing.", "llm", focus_id="q2", plan=plan
        ),
    )

    update = router_module.supervisor_node(_state(_plan("open", "open")))

    assert update["plan"] == plan
    assert update["research_focus_id"] == "q2"
    assert "(llm: Globex missing.)" in update["messages"][0].content
    assert calls == [
        {
            "event": "route_decision",
            "next": "research",
            "decided_by": "llm",
            "rationale": "Globex missing.",
            "focus": "Question 2?",
            "open_items": ["q2"],
        }
    ]


# --- one targeted search per item -------------------------------------------------


def _searched(plan: list[PlanItem], *ids: str) -> list[PlanItem]:
    return [{**item, "attempted": True} if item["id"] in ids else item for item in plan]


def test_an_open_item_that_was_already_searched_is_not_researchable() -> None:
    llm = _DecisionLLM(next="analytics")

    route = decide_route(_state(_searched(_plan("answered", "open"), "q2")), llm)  # type: ignore[arg-type]

    # Research is not offered; the model is still asked so q2 gets judged.
    assert route.next == "analytics"
    assert "Allowed next steps: analytics\n" in llm.prompt[1].content + "\n"


def test_a_hand_back_to_an_already_searched_item_moves_to_an_untried_one() -> None:
    llm = _DecisionLLM(next="research", focus_id="q1")

    route = decide_route(_state(_searched(_plan("open", "open"), "q1")), llm)  # type: ignore[arg-type]

    assert route.focus_id == "q2"


def test_the_plan_shows_judged_items_with_their_state() -> None:
    plan = _searched(_plan("answered", "open", "unanswerable", "open"), "q2")
    plan[0]["sources"] = ["acme.md"]
    llm = _DecisionLLM(next="research", focus_id="q4")

    decide_route(_state(plan), llm)  # type: ignore[arg-type]

    assert (
        "- q1 (answered from: acme.md) Question 1?\n"
        "- q2 (to judge; already searched) Question 2?\n"
        "- q3 (unanswerable from the knowledge base) Question 3?\n"
        "- q4 (to judge) Question 4?"
    ) in llm.prompt[1].content


def test_finding_snippets_are_long_enough_to_hold_a_chunk_s_figures() -> None:
    chunk = "Globex profile. " * 20 + "Typical ACV is $150K to $400K."
    state = _state(_plan("open"))
    state["research_findings"] = [{"source": "g.md", "content": chunk, "relevance_score": 1.0}]
    llm = _DecisionLLM(next="analytics")

    decide_route(state, llm)  # type: ignore[arg-type]

    assert "$150K to $400K" in llm.prompt[1].content


def test_with_one_step_left_open_items_are_still_judged() -> None:
    """The last targeted search must be judged: a forced step still asks the
    model, with its choice limited to that step."""

    llm = _DecisionLLM(next="analytics", coverage=[_cover("q2", "answered", "acme.md")])
    plan = [{**item, "attempted": True} for item in _plan("answered", "open")]

    route = decide_route(_state(plan), llm)  # type: ignore[arg-type]

    assert (route.next, route.decided_by) == ("analytics", "llm")
    assert route.plan is not None and route.plan[1]["status"] == "answered"
