"""Analytics Agent node — currently returns a mock metric."""

from typing import Any

from langchain_core.messages import AIMessage

from market_research_team.state import AgentState


def analytics_node(state: AgentState) -> dict[str, Any]:
    results = [
        {
            "metric": "research_finding_count",
            "value": float(len(state.get("research_findings", []))),
            "detail": "Number of research findings available for analysis (mock).",
        }
    ]
    return {
        "analytics_results": results,
        "messages": [
            AIMessage(content="Analytics complete (mock data).", name="analytics_agent")
        ],
    }
