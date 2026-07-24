"""Research Agent node: query rewriting -> vector retrieval -> cross-encoder reranking."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from market_research_team.agents.research.query_rewriter import rewrite_and_expand
from market_research_team.agents.research.reranker import load_cross_encoder, rerank
from market_research_team.agents.research.retriever import load_vectorstore, retrieve_for_queries
from market_research_team.config import settings
from market_research_team.state import AgentState, ResearchFinding

_RETRIEVAL_K_PER_QUERY = 4
_RERANK_TOP_N = 5


def run_research_pipeline(objective: str) -> tuple[list[ResearchFinding], int, int]:
    """Rewrite the objective into search queries, retrieve, then rerank.

    Returns the reranked findings plus the query and candidate counts, so
    the node can report them without recomputing anything.
    """

    llm = ChatAnthropic(model=settings.query_rewrite_model)
    queries = rewrite_and_expand(objective, llm)

    vectorstore = load_vectorstore(
        persist_dir=settings.vectorstore_dir,
        embedding_model_name=settings.embedding_model_name,
    )
    candidates = retrieve_for_queries(vectorstore, queries, k=_RETRIEVAL_K_PER_QUERY)

    cross_encoder = load_cross_encoder(settings.reranker_model_name)
    reranked = rerank(objective, candidates, cross_encoder, top_n=_RERANK_TOP_N)

    findings: list[ResearchFinding] = [
        {
            "source": document.metadata.get("source", "unknown"),
            "content": document.page_content,
            "relevance_score": score,
        }
        for document, score in reranked
    ]
    return findings, len(queries), len(candidates)


def research_node(state: AgentState) -> dict[str, Any]:
    findings, query_count, candidate_count = run_research_pipeline(state["objective"])

    return {
        "research_findings": findings,
        "messages": [
            AIMessage(
                content=(
                    f"Research complete: {query_count} queries, "
                    f"{candidate_count} candidates retrieved, "
                    f"{len(findings)} kept after reranking."
                ),
                name="research_agent",
            )
        ],
    }
