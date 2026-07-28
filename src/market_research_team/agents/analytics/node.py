"""Analytics Agent node: LLM tool-calling over native Python math/stat tools."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.config import settings
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding

_MAX_TOOL_ITERATIONS = 4

_SYSTEM_PROMPT = (
    "You are the Analytics Agent for a market and competitor research team. "
    "You are given research findings (free text) about competitors and the "
    "market, plus the research objective. Identify quantitative claims in "
    "the findings (prices, percentages, headcounts, growth rates, etc.) "
    "that are relevant to the objective, and use the provided tools to "
    "compute derived metrics (averages, differences, percent changes, "
    "growth rates, ranges) that help compare them. Call a tool for every "
    "number you report — never state a computed metric without a matching "
    "tool call. If the findings don't contain enough numeric data for a "
    "calculation, skip it rather than inventing numbers."
)


def _findings_to_context(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "No research findings are available."
    return "\n\n".join(f"[{finding['source']}] {finding['content']}" for finding in findings)


def run_tool_calling_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    objective: str,
    findings: list[ResearchFinding],
    *,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> list[AnalyticsResult]:
    """Drive an LLM tool-calling loop and record every tool call as a result.

    Every entry in the returned list corresponds to an actual tool
    invocation, so metrics stay grounded in a real computation rather than
    a number the LLM stated in free text. Tool errors (e.g. a percent
    change from a zero baseline) are reported back to the model as a
    ToolMessage instead of crashing the node, so a bad argument just costs
    a turn rather than the whole analytics step. `max_iterations` bounds
    the loop since nothing here is protected by LangGraph's own recursion
    limit — this is a plain Python loop, not a subgraph.
    """

    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research objective: {objective}\n\n"
                f"Findings:\n{_findings_to_context(findings)}"
            )
        ),
    ]

    results: list[AnalyticsResult] = []
    for _ in range(max_iterations):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            break

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = tools_by_name.get(tool_name)

            if tool is None:
                messages.append(
                    ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call["id"])
                )
                continue

            try:
                output = tool.invoke(tool_args)
            except Exception as exc:
                messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tool_call["id"]))
                continue

            results.append(
                {
                    "metric": tool_name,
                    "value": float(output),
                    "detail": f"{tool_name}({tool_args}) = {output}",
                }
            )
            messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

    return results


def run_analytics_pipeline(
    objective: str, findings: list[ResearchFinding]
) -> list[AnalyticsResult]:
    """Production wiring: the real Claude model bound to the native math/stat tools."""

    llm = ChatAnthropic(model=settings.query_rewrite_model)
    return run_tool_calling_loop(llm, ANALYTICS_TOOLS, objective, findings)


def analytics_node(state: AgentState) -> dict[str, Any]:
    results = run_analytics_pipeline(state["objective"], state.get("research_findings", []))

    return {
        "analytics_results": results,
        "messages": [
            AIMessage(
                content=f"Analytics complete: {len(results)} metric(s) computed via tool calls.",
                name="analytics_agent",
            )
        ],
    }
