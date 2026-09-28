"""Cache-aware wrapper around `retrieval.retriever.retrieve_for_queries`.

Keeps `retriever.py`'s core function pure and uncached; caching concerns
(serialization, keying, TTL/version invalidation) live here so they're all
testable in one place under `tests/caching/`.
"""

import time
from pathlib import Path

from langchain_core.documents import Document

from market_research_team.caching import stats, store
from market_research_team.caching.keys import retrieval_key
from market_research_team.config import settings
from market_research_team.observability import record_cache_event
from market_research_team.retrieval.retriever import retrieve_for_queries

_LAYER = "retrieval"


def _to_payload(documents: list[Document]) -> dict:
    return {
        "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    }


def _from_payload(payload: dict) -> list[Document]:
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in payload["documents"]
    ]


def get_cached_retrieval(
    queries: list[str], *, k: int, embedding_model_name: str, persist_dir: Path
) -> list[Document] | None:
    if not settings.cache_enabled:
        return None

    key_hash = retrieval_key(queries, k=k, embedding_model_name=embedding_model_name)
    start = time.perf_counter()
    try:
        conn = store.get_connection(settings.cache_db_path)
        version = store.current_vectorstore_version(persist_dir)
        payload = store.get(conn, layer=_LAYER, key_hash=key_hash, version=version)
    except Exception:
        payload = None
    latency_ms = (time.perf_counter() - start) * 1000

    hit = payload is not None
    stats.record(_LAYER, hit)
    record_cache_event(_LAYER, hit=hit, key_hash=key_hash, latency_ms=latency_ms)
    return _from_payload(payload) if payload is not None else None


def put_cached_retrieval(
    queries: list[str],
    documents: list[Document],
    *,
    k: int,
    embedding_model_name: str,
    persist_dir: Path,
) -> None:
    if not settings.cache_enabled:
        return

    key_hash = retrieval_key(queries, k=k, embedding_model_name=embedding_model_name)
    try:
        conn = store.get_connection(settings.cache_db_path)
        version = store.current_vectorstore_version(persist_dir)
        store.put(
            conn,
            layer=_LAYER,
            key_hash=key_hash,
            version=version,
            value=_to_payload(documents),
            ttl_seconds=settings.retrieval_cache_ttl_seconds,
        )
    except Exception:
        pass


def retrieve_for_queries_cached(
    vectorstore,
    queries: list[str],
    *,
    k: int = 4,
    embedding_model_name: str,
    persist_dir: Path,
) -> list[Document]:
    cached = get_cached_retrieval(
        queries, k=k, embedding_model_name=embedding_model_name, persist_dir=persist_dir
    )
    if cached is not None:
        return cached

    result = retrieve_for_queries(vectorstore, queries, k=k)
    put_cached_retrieval(
        queries, result, k=k, embedding_model_name=embedding_model_name, persist_dir=persist_dir
    )
    return result
