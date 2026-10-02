"""Cache-aware wrapper around `retrieval.reranker.rerank`.

Versioned on the vectorstore stamp like the retrieval cache: rerank scores
depend on which chunks were retrieved, which depend on the index.
"""

import time
from pathlib import Path

from langchain_core.documents import Document

from market_research_team.caching import stats, store
from market_research_team.caching.keys import rerank_key
from market_research_team.config import settings
from market_research_team.observability import record_cache_event
from market_research_team.retrieval.reranker import CrossEncoderModel, rerank

_LAYER = "rerank"


def _chunk_id(document: Document) -> str:
    return document.metadata.get("chunk_id", document.page_content)


def _to_payload(scored: list[tuple[Document, float]]) -> dict:
    return {
        "scored": [
            {"page_content": d.page_content, "metadata": d.metadata, "score": s} for d, s in scored
        ]
    }


def _from_payload(payload: dict) -> list[tuple[Document, float]]:
    return [
        (Document(page_content=e["page_content"], metadata=e["metadata"]), e["score"])
        for e in payload["scored"]
    ]


def rerank_cached(
    objective: str,
    documents: list[Document],
    model: CrossEncoderModel,
    *,
    top_n: int = 5,
    score_floor: float | None = None,
    reranker_model_name: str,
    persist_dir: Path,
) -> list[tuple[Document, float]]:
    if not settings.cache_enabled:
        return rerank(objective, documents, model, top_n=top_n, score_floor=score_floor)

    chunk_ids = [_chunk_id(d) for d in documents]
    key_hash = rerank_key(
        objective,
        chunk_ids,
        reranker_model_name=reranker_model_name,
        top_n=top_n,
        score_floor=score_floor,
    )

    start = time.perf_counter()
    try:
        conn = store.get_connection(settings.cache_db_path)
        version = store.cache_version(persist_dir)
        payload = store.get(conn, layer=_LAYER, key_hash=key_hash, version=version)
    except Exception:
        payload = None
    latency_ms = (time.perf_counter() - start) * 1000

    hit = payload is not None
    stats.record(_LAYER, hit)
    record_cache_event(_LAYER, hit=hit, key_hash=key_hash, latency_ms=latency_ms)
    if hit:
        return _from_payload(payload)

    result = rerank(objective, documents, model, top_n=top_n, score_floor=score_floor)
    try:
        store.put(
            conn,
            layer=_LAYER,
            key_hash=key_hash,
            version=version,
            value=_to_payload(result),
            ttl_seconds=settings.retrieval_cache_ttl_seconds,
        )
    except Exception:
        pass
    return result
