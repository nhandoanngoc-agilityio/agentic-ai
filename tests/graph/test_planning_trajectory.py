"""Trajectories through the real graph with the real supervisor logic
(`decide_route`): only the model's answers are scripted.

Plan -> broad research -> hand-back for the uncovered sub-question -> analysis
-> report, with every decision explained; and a sub-question the knowledge
base can't answer becomes unanswerable, stops research and is named in the
report.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.planner import node as planner_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as router_module
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import PlanItem, ResearchFinding, new_run_state

_ACME: ResearchFinding = {
    "source": "acme.md",
    "content": "Acme Starter is $49 per seat.",
    "relevance_score": 2.0,
}
_GLOBEX: ResearchFinding = {
    "source": "globex.md",
    "content": "Globex ACV is $150K to $400K.",
    "relevance_score": 1.5,
}
_PLAN: list[PlanItem] = [
    {"id": "q1", "question": "What does Acme charge per seat?", "status": "open", "sources": []},
    {"id": "q2", "question": "What is Globex's contract value?", "status": "open", "sources": []},
]


def _unheld(findings: list[ResearchFinding], exclude: frozenset) -> list[ResearchFinding]:
    """Like the real pipeline: findings already held are not returned again."""

    return [f for f in findings if (f["source"], f["content"]) not in exclude]


def _cover(item_id: str, status: str, *sources: str) -> SimpleNamespace:
    return SimpleNamespace(id=item_id, status=status, sources=list(sources))


def _decision(next_step: str, rationale: str, *coverage: SimpleNamespace, focus_id=None):
    return SimpleNamespace(
        next=next_step, rationale=rationale, coverage=list(coverage), focus_id=focus_id
    )


class _ScriptedSupervisorModel:
    """Stands in for the chat model inside `decide_route`: one scripted
    structured answer per LLM-decided visit, in order."""

    def __init__(self, decisions: list[SimpleNamespace]) -> None:
        self._decisions = list(decisions)

    def with_structured_output(self, _schema: object) -> "_ScriptedSupervisorModel":
        return self

    def invoke(self, _messages: list[Any], config: object = None) -> SimpleNamespace:
        return self._decisions.pop(0)


@pytest.fixture
def audit_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        router_module.audit,
        "record",
        lambda event, thread_id, **fields: events.append({"event": event, **fields}),
    )
    return events


@pytest.fixture(autouse=True)
def _stub_everything_but_the_supervisor_logic(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    research_calls: list[Any] = []

    def _research(objective: str, focus: str | None = None, exclude=frozenset()):
        research_calls.append(focus)
        # The broad pass finds Acme only; a pass aimed at Globex finds Globex.
        found = [_GLOBEX] if focus and "Globex" in focus else [_ACME]
        return _unheld(found, exclude), 2, 4, []

    async def _write(filename: str, content: str) -> str:
        return f"reports/{filename}"

    monkeypatch.setattr(planner_node_module, "run_planner", lambda objective: _PLAN)
    monkeypatch.setattr(research_node_module, "run_research_pipeline", _research)
    monkeypatch.setattr(
        analytics_node_module,
        "run_analytics_pipeline",
        lambda objective, findings: [
            {"metric": "mean", "value": 49.0, "detail": "mean", "entity": None, "inputs": [49.0]}
        ],
    )
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(
        reporting_node_module, "draft_report", lambda *args, **kwargs: "# Pricing report"
    )
    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _write)
    return research_calls


def _run(
    decisions: list[SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    audit_events: list[dict[str, Any]],
    thread: str,
):
    model = _ScriptedSupervisorModel(decisions)
    monkeypatch.setattr(router_module, "get_chat_model", lambda: model)
    graph = build_production_graph(InMemorySaver())
    paused = run_graph(
        new_run_state("Compare Acme and Globex pricing"), compiled_graph=graph, thread_id=thread
    )
    draft = cast(dict[str, Any], paused)["__interrupt__"][0].value["content"]
    final = run_graph(Command(resume={"approved": True}), compiled_graph=graph, thread_id=thread)
    # Every scripted answer used, none missing: running out would surface as a
    # silent routing fallback rather than a failure.
    assert model._decisions == []
    assert not [e for e in audit_events if e.get("component") == "supervisor_router"]
    return final, draft


def test_plan_drives_a_targeted_hand_back_until_every_item_is_answered(
    monkeypatch: pytest.MonkeyPatch,
    audit_events: list[dict[str, Any]],
    _stub_everything_but_the_supervisor_logic: list[Any],
) -> None:
    final, draft = _run(
        [
            _decision(
                "research",
                "Acme is covered; Globex contract value is missing.",
                _cover("q1", "answered", "acme.md"),
                _cover("q2", "open"),
                focus_id="q2",
            ),
            # q2 had its one targeted search, so analytics is the only step
            # left; the model is still asked, and judges q2 answered.
            _decision(
                "analytics",
                "Both questions are answered; analyze them.",
                _cover("q1", "answered", "acme.md"),
                _cover("q2", "answered", "globex.md"),
            ),
        ],
        monkeypatch,
        audit_events,
        "plan-covered",
    )

    # Research ran twice: broadly, then aimed at the open sub-question.
    assert _stub_everything_but_the_supervisor_logic == [None, "What is Globex's contract value?"]
    assert [(item["id"], item["status"]) for item in final.get("plan", [])] == [
        ("q1", "answered"),
        ("q2", "answered"),
    ]
    assert final["report_path"] and not final.get("error")
    assert "Open questions" not in draft

    decisions = [e for e in audit_events if e["event"] == "route_decision"]
    assert [(d["next"], d["decided_by"]) for d in decisions] == [
        ("research", "rule"),
        ("research", "llm"),
        ("analytics", "llm"),  # the only step left, asked so q2 gets judged
        ("reporting", "rule"),  # analysis is current and the plan is resolved
        ("FINISH", "rule"),
    ]
    assert all(d["rationale"] for d in decisions)
    (finished,) = [e for e in audit_events if e["event"] == "run_finished"]
    assert finished["plan_coverage"] == {"q1": "answered", "q2": "answered"}


def test_an_unanswerable_item_stops_research_and_is_named_in_the_report(
    monkeypatch: pytest.MonkeyPatch,
    audit_events: list[dict[str, Any]],
    _stub_everything_but_the_supervisor_logic: list[Any],
) -> None:
    # The knowledge base has nothing on q2: the targeted pass returns Acme again.
    monkeypatch.setattr(
        research_node_module,
        "run_research_pipeline",
        lambda objective, focus=None, exclude=frozenset(): (_unheld([_ACME], exclude), 2, 4, []),
    )

    final, draft = _run(
        [
            _decision(
                "research",
                "Globex is missing.",
                _cover("q1", "answered", "acme.md"),
                focus_id="q2",
            ),
        ],
        monkeypatch,
        audit_events,
        "plan-gap",
    )

    assert [(item["id"], item["status"]) for item in final.get("plan", [])] == [
        ("q1", "answered"),
        ("q2", "unanswerable"),
    ]
    assert "## Open questions" in draft
    assert "- What is Globex's contract value?" in draft
    decisions = [e for e in audit_events if e["event"] == "route_decision"]
    assert [(d["next"], d["decided_by"]) for d in decisions] == [
        ("research", "rule"),
        ("research", "llm"),
        ("analytics", "rule"),  # research no longer offered: q2 is unanswerable
        ("reporting", "rule"),  # analysis is current and the plan is resolved
        ("FINISH", "rule"),
    ]
