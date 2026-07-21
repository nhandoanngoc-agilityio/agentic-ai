"""Reporting Agent node — currently records a mock report path."""

from typing import Any

from langchain_core.messages import AIMessage

from market_research_team.state import AgentState


def reporting_node(state: AgentState) -> dict[str, Any]:
    return {
        "report_path": "reports/mock-report.md",
        "messages": [
            AIMessage(
                content="Report complete (mock — no file written yet).",
                name="reporting_agent",
            )
        ],
    }
