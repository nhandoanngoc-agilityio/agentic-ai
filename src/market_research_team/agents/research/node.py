"""Research Agent node — currently returns mock findings."""

from typing import Any

from langchain_core.messages import AIMessage

from market_research_team.state import AgentState


def research_node(state: AgentState) -> dict[str, Any]:
    findings = [
        {
            "source": "mock://research-agent",
            "content": f"Mock research finding for objective: {state['objective']!r}",
            "relevance_score": 0.0,
        }
    ]
    return {
        "research_findings": findings,
        "messages": [
            AIMessage(content="Research complete (mock data).", name="research_agent")
        ],
    }
