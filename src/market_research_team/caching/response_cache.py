"""Opt-in full-pipeline response cache: skips re-running research/analytics
for an exact repeated objective.

Deliberately narrow in what it caches and how: exact objective string match
only (no semantic similarity -- see the design discussion), and it only ever
short-circuits `research`/`analytics`. `reporting_node`'s draft, output
guardrails, and human-approval `interrupt()` always run fresh and are never
skipped -- a cache hit changes what feeds into the report, never whether a
human reviews it.
"""

import time
from pathlib import Path

from market_research_team.caching import stats, store
from market_research_team.caching.keys import response_key
from market_research_team.config import settings
from market_research_team.observability import record_cache_event
from market_research_team.state import AnalyticsResult, ResearchFinding

_LAYER = "response"


def get_cached_response(
    objective: str, persist_dir: Path
) -> tuple[list[ResearchFinding], list[AnalyticsResult]] | None:
    if not settings.response_cache_enabled:
        return None

    key_hash = response_key(objective)
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
    if not hit:
        return None
    return payload["research_findings"], payload["analytics_results"]


def put_cached_response(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
    persist_dir: Path,
) -> None:
    if not settings.response_cache_enabled:
        return

    key_hash = response_key(objective)
    try:
        conn = store.get_connection(settings.cache_db_path)
        version = store.current_vectorstore_version(persist_dir)
        store.put(
            conn,
            layer=_LAYER,
            key_hash=key_hash,
            version=version,
            value={"research_findings": findings, "analytics_results": results},
            ttl_seconds=settings.response_cache_ttl_seconds,
        )
    except Exception:
        pass
