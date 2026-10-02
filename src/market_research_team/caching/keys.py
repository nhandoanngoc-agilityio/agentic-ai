"""Deterministic cache-key construction for the retrieval, rerank, and
response caches."""

import hashlib
import json

from market_research_team.config import settings
from market_research_team.security.patterns import INJECTION_PATTERNS
from market_research_team.versioning import instructions_fingerprint


def hash_key(*parts: object) -> str:
    """Sha256 hex digest of a JSON-normalized tuple of parts, prefixed with
    `cache_namespace` so tenants/environments sharing a cache file stay
    isolated."""

    blob = json.dumps((settings.cache_namespace, *parts), sort_keys=True, default=str)
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


def _active_model() -> str:
    if settings.llm_provider == "openai":
        return settings.openai_model
    return settings.anthropic_model


def _injection_policy_fingerprint() -> str:
    return hash_key(
        "injection_policy", [(name, regex.pattern) for name, regex in INJECTION_PATTERNS]
    )


def response_key(objective: str) -> str:
    """Exact-match on the objective, deliberately -- see caching/response_cache.py.

    Also keyed on the policy that produced the cached value: the LLM
    provider/model and system prompts (analytics results depend on them;
    see `versioning.instructions_fingerprint`) and the injection
    patterns (cached findings already passed that filter). Changing either
    makes old entries unreachable instead of serving them under new rules.
    """

    return hash_key(
        "response",
        objective,
        settings.llm_provider,
        _active_model(),
        instructions_fingerprint(),
        _injection_policy_fingerprint(),
    )
