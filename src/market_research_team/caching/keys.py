"""Deterministic cache-key construction for the retrieval, rerank, and
response caches."""

import hashlib
import json


def hash_key(*parts: object) -> str:
    """Sha256 hex digest of a JSON-normalized tuple of parts."""

    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def retrieval_key(queries: list[str], *, k: int, embedding_model_name: str) -> str:
    """Keyed on the post-rewrite query list, not the raw objective -- query
    rewriting is itself LLM-based and non-deterministic, so caching upstream
    of it would rarely hit."""

    return hash_key("retrieval", sorted(queries), k, embedding_model_name)


def rerank_key(
    objective: str,
    chunk_ids: list[str],
    *,
    reranker_model_name: str,
    top_n: int,
    score_floor: float | None,
) -> str:
    """Keyed on the candidate set's chunk_ids (order-independent), since a
    chunk_id already uniquely identifies content in the seeded vectorstore."""

    return hash_key("rerank", objective, sorted(chunk_ids), reranker_model_name, top_n, score_floor)


def response_key(objective: str) -> str:
    """Exact-match only, deliberately -- see caching/response_cache.py."""

    return hash_key("response", objective)
