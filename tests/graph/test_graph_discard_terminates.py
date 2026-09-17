"""Full-graph regression test for the comparison UI's "discard" outcome.

The existing interrupt tests (tests/test_reporting_node_interrupt.py) build an
ad-hoc `START -> reporting -> END` single-node graph, so they can never see what
the supervisor does *after* `reporting_node` returns. The real graph is cyclic
(`supervisor -> reporting -> supervisor -> ...`), and a discard that sets neither
`report_path` nor `error` used to send the supervisor straight back to Reporting,
which redrafted and interrupted again -- forever.

So this test runs the *real* compiled graph with the *real* `decide_next_step`
(only the LLM/MCP/pipeline I/O is faked) and asserts a discard terminates the run
without a second interrupt.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_research_team.agents.analytics import node as analytics_node_module
from market_research_team.agents.reporting import node as reporting_node_module
from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as supervisor_router_module
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding


def _fake_research_pipeline(objective: str) -> tuple[list[ResearchFinding], int, int]:
    return [{"source": "mock", "content": f"finding for {objective}", "relevance_score": 0.9}], 1, 1


def _fake_analytics_pipeline(
    objective: str, findings: list[ResearchFinding]
) -> list[AnalyticsResult]:
    return [{"metric": "count", "value": float(len(findings)), "detail": "d"}]


def _fake_draft_report(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
    llm: Any,
    feedback: str | None = None,
) -> str:
    return "# Mock Report"


async def _fake_write_report_via_mcp(filename: str, content: str) -> str:
    raise AssertionError("a discarded report must never be written to disk")


class _FakeStructuredRouteLLM:
    """Stands in for the supervisor's chat model so the *real* `decide_next_step`
    runs unmodified. It reads the "Allowed next steps:" line the router builds and
    picks the last option, which walks research -> analytics -> reporting."""

    def with_structured_output(self, schema: Any) -> "_FakeStructuredRouteLLM":
        self._schema = schema
        return self

    def invoke(self, messages: list[Any]) -> Any:
        context = str(messages[-1].content)
        allowed_line = next(
            line for line in context.splitlines() if line.startswith("Allowed next steps:")
        )
        allowed = [option.strip() for option in allowed_line.split(":", 1)[1].split(",")]
        return self._schema(next=allowed[-1])


@pytest.fixture(autouse=True)
def _stub_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research_node_module, "run_research_pipeline", _fake_research_pipeline)
    monkeypatch.setattr(analytics_node_module, "run_analytics_pipeline", _fake_analytics_pipeline)
    monkeypatch.setattr(reporting_node_module, "get_chat_model", lambda: None)
    monkeypatch.setattr(reporting_node_module, "draft_report", _fake_draft_report)
    monkeypatch.setattr(reporting_node_module, "write_report_via_mcp", _fake_write_report_via_mcp)
    # NOTE: `run_supervisor_decision` is deliberately *not* stubbed here (unlike
    # test_checkpointing.py / test_run_graph_safety.py) -- the routing decision is
    # exactly what's under test. Only its LLM call is faked.
    monkeypatch.setattr(supervisor_router_module, "get_chat_model", _FakeStructuredRouteLLM)


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def test_discard_terminates_the_real_graph_without_a_second_interrupt() -> None:
    compiled_graph = build_production_graph(InMemorySaver())
    thread_id = "discard-real-graph"

    paused = run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id=thread_id)

    assert "__interrupt__" in paused, "expected the reporting node to pause for review"
    assert paused["__interrupt__"][0].value["action"] == "write_report"

    result = run_graph(
        Command(resume={"approved": False, "discard": True}),
        compiled_graph=compiled_graph,
        thread_id=thread_id,
    )

    # The whole point: the supervisor must not send the run back to Reporting to
    # redraft, which would surface as a second pause here.
    assert "__interrupt__" not in result, "a discarded run must not interrupt again"
    assert result["next"] == "FINISH"
    assert result.get("report_discarded") is True
    assert result.get("report_path") is None
    assert result.get("error") is None

    # And the run is genuinely finished -- nothing is left pending on the thread.
    snapshot = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snapshot.next == ()
    assert snapshot.interrupts == ()


def test_approval_still_completes_the_real_graph_with_the_same_wiring() -> None:
    """Control case: the discard early-return must not short-circuit a normal
    approved run through the same real-router graph."""

    async def _write(filename: str, content: str) -> str:
        return f"reports/{filename}"

    compiled_graph = build_production_graph(InMemorySaver())
    thread_id = "approve-real-graph"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reporting_node_module, "write_report_via_mcp", _write)

        paused = run_graph(_initial_state(), compiled_graph=compiled_graph, thread_id=thread_id)
        assert "__interrupt__" in paused

        result = run_graph(
            Command(resume={"approved": True}),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
        )

    assert "__interrupt__" not in result
    assert result["next"] == "FINISH"
    assert result["report_path"] == "reports/assess-competitor-pricing-strategy.md"
    assert result.get("report_discarded") is None
