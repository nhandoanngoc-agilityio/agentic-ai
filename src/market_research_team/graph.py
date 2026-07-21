"""Top-level graph assembly.
"""

from langgraph.graph import END, START, StateGraph

from market_research_team.agents.analytics.node import analytics_node
from market_research_team.agents.reporting.node import reporting_node
from market_research_team.agents.research.node import research_node
from market_research_team.state import AgentState
from market_research_team.supervisor.router import route_from_supervisor, supervisor_node

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_node)
builder.add_node("analytics", analytics_node)
builder.add_node("reporting", reporting_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "analytics": "analytics",
        "reporting": "reporting",
        END: END,
    },
)
builder.add_edge("research", "supervisor")
builder.add_edge("analytics", "supervisor")
builder.add_edge("reporting", "supervisor")

graph = builder.compile()
