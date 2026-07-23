"""Query rewriting/expansion: turns one research objective into several search queries."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

_SYSTEM_PROMPT = (
    "You are a search query planner for a market and competitor research "
    "assistant backed by a vector database of competitor profiles and "
    "market-overview documents. Given a research objective, produce "
    "between 2 and 4 distinct search queries that together cover the "
    "objective from different angles (e.g. pricing, strengths/weaknesses, "
    "recent moves, market context). Keep each query short and specific — "
    "phrase them the way the terms would actually appear in the source "
    "documents, not as questions."
)


class _QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="2-4 distinct, keyword-style search queries covering the objective."
    )


def rewrite_and_expand(objective: str, llm: BaseChatModel) -> list[str]:
    """Expand a research objective into several vector-search queries.

    Falls back to the raw objective as a single query if the LLM call
    fails or returns nothing usable, so a broken rewriter degrades the
    retrieval pipeline instead of breaking it outright.
    """

    try:
        structured_llm = llm.with_structured_output(_QueryExpansion)
        result = structured_llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=objective)]
        )
        queries = [query.strip() for query in result.queries if query.strip()]
    except Exception:
        return [objective]

    return queries or [objective]
