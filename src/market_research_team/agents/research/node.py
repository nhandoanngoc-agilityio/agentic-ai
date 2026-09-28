"""Research Agent node: query rewriting -> vector retrieval -> cross-encoder reranking."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from market_research_team.caching.memo import cached_cross_encoder
from market_research_team.caching.rerank_cache import rerank_cached
from market_research_team.caching.retrieval_cache import retrieve_for_queries_cached
from market_research_team.config import settings
from market_research_team.llm import get_chat_model
from market_research_team.retrieval.query_rewriter import rewrite_and_expand
from market_research_team.retrieval.retriever import load_vectorstore
from market_research_team.security.patterns import INJECTION_PATTERNS, find_matches
from market_research_team.state import AgentState, GuardrailEvent, ResearchFinding

_RETRIEVAL_K_PER_QUERY = 4
_RERANK_TOP_N = 5


def filter_injected_chunks(
    reranked: list[tuple[Document, float]],
) -> tuple[list[tuple[Document, float]], list[GuardrailEvent]]:
    """Retrieval guardrail: drop chunks that contain prompt-injection text.

    The ingestion pipeline accepts any file placed in `data/raw/`, so a
    poisoned document is the realistic way instructions reach the report
    prompt. The reporting prompt already fences chunks as data; this removes
    the obvious cases before they get that far, and records each drop.
    """

    kept: list[tuple[Document, float]] = []
    events: list[GuardrailEvent] = []
    for document, score in reranked:
        hits = find_matches(document.page_content, INJECTION_PATTERNS)
        if hits:
            source = document.metadata.get("source", "unknown")
            events.append(
                {
                    "layer": "retrieval",
                    "rule": "injection_in_chunk",
                    "detail": f"dropped chunk from {source}: {', '.join(hits)}",
                }
            )
            continue
        kept.append((document, score))
    return kept, events


def run_research_pipeline(
    objective: str,
) -> tuple[list[ResearchFinding], int, int, list[GuardrailEvent]]:
    """Rewrite the objective into search queries, retrieve, then rerank.

    Returns the reranked findings plus the query and candidate counts and
    any retrieval guardrail events, so the node can report them without
    recomputing anything.
    """

    llm = get_chat_model()
    queries = rewrite_and_expand(objective, llm)

    vectorstore = load_vectorstore(
        persist_dir=settings.vectorstore_dir,
        embedding_model_name=settings.embedding_model_name,
    )
    candidates = retrieve_for_queries_cached(
        vectorstore,
        queries,
        k=_RETRIEVAL_K_PER_QUERY,
        embedding_model_name=settings.embedding_model_name,
        persist_dir=settings.vectorstore_dir,
    )

    cross_encoder = cached_cross_encoder(settings.reranker_model_name)
    reranked = rerank_cached(
        objective,
        candidates,
        cross_encoder,
        top_n=_RERANK_TOP_N,
        score_floor=settings.rerank_score_floor,
        reranker_model_name=settings.reranker_model_name,
        persist_dir=settings.vectorstore_dir,
    )
    reranked, events = filter_injected_chunks(reranked)

    findings: list[ResearchFinding] = [
        {
            "source": document.metadata.get("source", "unknown"),
            "content": document.page_content,
            "relevance_score": score,
        }
        for document, score in reranked
    ]
    return findings, len(queries), len(candidates), events


def research_node(state: AgentState) -> dict[str, Any]:
    findings, query_count, candidate_count, events = run_research_pipeline(state["objective"])

    return {
        "research_findings": findings,
        "guardrail_events": events,
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
