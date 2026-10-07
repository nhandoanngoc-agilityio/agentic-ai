"""`run_graph(on_step=...)` streams the real graph to report progress, and must
return exactly what the plain `invoke` path returns -- including the
`__interrupt__` a paused run carries -- so the UI can switch to it safely."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as supervisor_router_module
from market_research_team.agents.supervisor.router import SupervisorRoute
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState


def _route(state: AgentState) -> SupervisorRoute:
    if not state.get("research_findings"):
        return SupervisorRoute("research")
    if not state.get("analytics_results"):
        return SupervisorRoute("analytics")
    if not state.get("report_path"):
        return SupervisorRoute("reporting")
    return SupervisorRoute("FINISH")


async def _write(filename: str, content: str) -> str:
    return f"reports/{filename}"


@pytest.fixture(autouse=True)
def _stub_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    finding = {"source": "acme.md", "content": "Acme charges $49 per seat.", "relevance_score": 1.0}
    monkeypatch.setattr(
        research_node_module,
        "run_research_pipeline",
        lambda objective, focus=None, exclude=frozenset(): ([finding], 2, 4, []),
    )
    monkeypatch.setattr(
        analytics_node_module,
        "run_analytics_pipeline",
        lambda objective, findings: [{"metric": "mean", "value": 49.0, "detail": "d"}],
    )
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(
        reporting_node_module,
        "draft_report",
        lambda objective, findings, results, llm, feedback=None: "# Report",
    )
    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _write)
    monkeypatch.setattr(supervisor_router_module, "run_supervisor_decision", _route)


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess Acme pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def _comparable(state: dict[str, Any]) -> dict[str, Any]:
    """State with messages reduced to (name, content), since message ids differ per run."""

    out = {k: v for k, v in state.items() if k not in ("messages", "__interrupt__")}
    out["messages"] = [(m.name, m.content) for m in state["messages"]]
    if "__interrupt__" in state:
        out["__interrupt__"] = [i.value for i in state["__interrupt__"]]
    return out


def _run_both(command: Any, steps: list[str], thread: str) -> tuple[Any, Any]:
    plain_graph = build_production_graph(InMemorySaver())
    streamed_graph = build_production_graph(InMemorySaver())
    plain = run_graph(_initial_state(), compiled_graph=plain_graph, thread_id=thread)
    streamed = run_graph(
        _initial_state(),
        compiled_graph=streamed_graph,
        thread_id=thread,
        on_step=lambda node, update: steps.append(node),
    )
    if command is None:
        return plain, streamed
    plain = run_graph(command, compiled_graph=plain_graph, thread_id=thread)
    streamed = run_graph(
        command,
        compiled_graph=streamed_graph,
        thread_id=thread,
        on_step=lambda node, update: steps.append(node),
    )
    return plain, streamed


def test_streamed_run_pauses_with_the_same_state_and_interrupt_as_invoke() -> None:
    steps: list[str] = []

    plain, streamed = _run_both(None, steps, "t-pause")

    assert "__interrupt__" in streamed
    assert _comparable(streamed) == _comparable(plain)
    assert steps == [
        "input_guard",
        "planner",
        "supervisor",
        "research",
        "supervisor",
        "analytics",
        "supervisor",
        "reporting",
    ]


def test_streamed_resume_finishes_with_the_same_state_as_invoke() -> None:
    steps: list[str] = []

    plain, streamed = _run_both(Command(resume={"approved": True}), steps, "t-finish")

    assert streamed["report_path"] == "reports/assess-acme-pricing-strategy.md"
    assert "__interrupt__" not in streamed
    assert _comparable(streamed) == _comparable(plain)
    assert steps[-2:] == ["report_review", "supervisor"]


def test_a_failing_step_callback_does_not_break_the_run() -> None:
    def _boom(node: str, update: dict[str, Any]) -> None:
        raise RuntimeError("display bug")

    result = run_graph(
        _initial_state(),
        compiled_graph=build_production_graph(InMemorySaver()),
        thread_id="t-boom",
        on_step=_boom,
    )

    assert "__interrupt__" in result
    assert result.get("error") is None
