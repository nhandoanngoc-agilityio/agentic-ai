"""Reporting Agent node: drafts a markdown report and writes it via the
local MCP filesystem server."""

import asyncio
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from market_research_team.agents.reporting.mcp_client import load_reporting_tools
from market_research_team.config import settings
from market_research_team.state import AgentState, AnalyticsResult, ResearchFinding

_SYSTEM_PROMPT = (
    "You write concise markdown research reports for a market and "
    "competitor research team. Given the objective, research findings, "
    "and computed analytics, write a well-organized markdown report with "
    "headings for Objective, Key Findings, Analysis, and Recommendation. "
    "Only use the information provided — do not invent facts, figures, or "
    "sources that weren't given to you."
)


def _findings_section(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "No research findings were available."
    return "\n".join(f"- ({finding['source']}) {finding['content']}" for finding in findings)


def _results_section(results: list[AnalyticsResult]) -> str:
    if not results:
        return "No analytics were computed."
    return "\n".join(
        f"- {result['metric']}: {result['value']} ({result['detail']})" for result in results
    )


def _fallback_report(
    objective: str, findings: list[ResearchFinding], results: list[AnalyticsResult]
) -> str:
    """Deterministic template used if the LLM call fails, so a broken draft
    step still produces a usable (if unpolished) report."""

    return (
        f"# Research Report\n\n"
        f"## Objective\n{objective}\n\n"
        f"## Key Findings\n{_findings_section(findings)}\n\n"
        f"## Analysis\n{_results_section(results)}\n"
    )


def draft_report(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
    llm: BaseChatModel,
) -> str:
    """Draft the markdown report body, falling back to a plain template on LLM failure."""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Objective: {objective}\n\n"
                        f"Findings:\n{_findings_section(findings)}\n\n"
                        f"Analytics:\n{_results_section(results)}"
                    )
                ),
            ]
        )
        content = response.content
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass

    return _fallback_report(objective, findings, results)


def _slugify(objective: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in objective)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60] or "report"


def _extract_tool_text(result: object) -> str:
    """MCP tool results come back as a list of content blocks; join their text."""

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(block.get("text", "") for block in result if isinstance(block, dict))
    return str(result)


async def run_reporting_pipeline(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
) -> str:
    """Draft the report and write it via the local MCP server. Returns the written path."""

    llm = ChatAnthropic(model=settings.query_rewrite_model)
    report_markdown = draft_report(objective, findings, results, llm)

    tools = await load_reporting_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_report")

    filename = f"{_slugify(objective)}.md"
    raw_result = await write_tool.ainvoke({"filename": filename, "content": report_markdown})
    return _extract_tool_text(raw_result)


def reporting_node(state: AgentState) -> dict[str, Any]:
    report_path = asyncio.run(
        run_reporting_pipeline(
            state["objective"],
            state.get("research_findings", []),
            state.get("analytics_results", []),
        )
    )

    return {
        "report_path": report_path,
        "messages": [
            AIMessage(content=f"Report written to {report_path}.", name="reporting_agent")
        ],
    }
