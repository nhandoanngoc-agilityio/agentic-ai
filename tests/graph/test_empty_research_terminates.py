"""Full-graph regression: an objective the knowledge base can't answer (every
chunk below the rerank floor) used to bounce supervisor -> research until the
recursion limit, because `decide_next_step` always sends a run with no findings
back to Research. Now the first empty pass ends the run with a clear error."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.supervisor import router as supervisor_router_module
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState


def test_off_topic_objective_ends_after_one_research_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def _empty_pipeline(objective: str, focus: str | None = None):
        calls.append(objective)
        return [], 3, 0, []

    monkeypatch.setattr(research_node_module, "run_research_pipeline", _empty_pipeline)
    # The real `decide_route` runs; it must finish on `error` without an LLM call.
    monkeypatch.setattr(supervisor_router_module, "get_chat_model", lambda: None)

    initial: AgentState = {
        "messages": [],
        "objective": "Recommend a sourdough starter feeding schedule",
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }
    result = run_graph(
        initial, compiled_graph=build_production_graph(InMemorySaver()), thread_id="off-topic"
    )

    assert calls == ["Recommend a sourdough starter feeding schedule"]
    assert "no relevant material" in result["error"]
    assert "Recursion limit" not in result["error"]
    assert result.get("report_path") is None
