"""Agent version manifest: one fingerprint for everything that shapes behaviour.

An agent's behaviour depends on more than its code release. It also depends on
its prompts, model and parameters, tool schemas, knowledge index and safety
limits. `build_manifest()` collects each of those into a component, hashes each
component, and hashes the hashes into one `fingerprint`. The version label
`agent_version()` is `<package release>+<fingerprint>`, so a prompt edit or a
vectorstore reseed yields a new, comparable version without anyone remembering
to bump a number.

The label is stamped on Langfuse traces (`graph.run_graph`), every audit-log
line (`security/audit.py`) and saved eval results
(`evaluation/offline_eval.py`), so any trace, rating or eval score can be tied
back to the exact agent definition that produced it.

Agent modules are imported lazily inside the builders: `security/audit.py`
calls into this module, and the agent nodes import `llm` -> `observability` ->
`audit`, so a module-level import here would be circular.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from importlib import metadata
from typing import Any

from market_research_team.config import settings

_PACKAGE_NAME = "market-research-analyst-team"
_HASH_LENGTH = 12


def _digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_HASH_LENGTH]


def release() -> str:
    """The package version from `pyproject.toml` -- the human-chosen part of the label."""

    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0"


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Commit the running code was built from: CI env first (containers have no
    `.git`), then the local checkout, else `unknown`. Build provenance only --
    not part of the fingerprint, which tracks behaviour, not commits."""

    for var in ("GIT_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
        if value := os.environ.get(var):
            return value[:_HASH_LENGTH]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _instructions() -> dict[str, str]:
    from market_research_team.agents.analytics import node as analytics
    from market_research_team.agents.planner import node as planner
    from market_research_team.agents.reporting import node as reporting
    from market_research_team.agents.supervisor import router
    from market_research_team.retrieval import query_rewriter

    return {
        "planner": _digest(planner.SYSTEM_PROMPT),
        "reporting_guidance": _digest(reporting.GUIDANCE_INTRO),
        "supervisor": _digest(router.SYSTEM_PROMPT),
        "query_rewriter": _digest(query_rewriter.SYSTEM_PROMPT),
        "analytics": _digest(analytics.SYSTEM_PROMPT),
        "reporting": _digest(reporting.SYSTEM_PROMPT),
    }


def _model() -> dict[str, Any]:
    model_id = (
        settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model
    )
    return {
        "provider": settings.llm_provider,
        "model": model_id,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


def _tools() -> dict[str, Any]:
    """Schema hash per tool, grouped by the agent allowed to call it -- the
    grouping is the permission model (analytics can't write files, reporting
    can't compute)."""

    from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
    from market_research_team.mcp_server import fs_server

    analytics = {tool.name: _digest(tool.args) for tool in ANALYTICS_TOOLS}
    mcp_tools = {
        tool.name: _digest(tool.parameters)
        for tool in fs_server.mcp_server._tool_manager.list_tools()
    }
    return {
        "analytics": analytics,
        "mcp_reports": mcp_tools,
        "mcp_limits": {
            "max_filename_length": fs_server._MAX_FILENAME_LENGTH,
            "max_report_bytes": fs_server._MAX_REPORT_BYTES,
        },
    }


def _knowledge_static() -> dict[str, Any]:
    from market_research_team.agents.research import node as research
    from market_research_team.retrieval import evidence

    return {
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "rerank_score_floor": settings.rerank_score_floor,
        "section_chunk_size": settings.section_chunk_size,
        "leaf_chunk_size": settings.leaf_chunk_size,
        "leaf_chunk_overlap": settings.leaf_chunk_overlap,
        "retrieval_k_per_query": research._RETRIEVAL_K_PER_QUERY,
        "rerank_top_n": research._RERANK_TOP_N,
        "max_research_findings": settings.run_policy.max_findings,
        "topic_keywords": _digest(evidence.TOPIC_KEYWORDS),
    }


def _memory_safety() -> dict[str, Any]:
    from market_research_team.agents.supervisor import router
    from market_research_team.security import input_validation, output_filters, patterns

    return {
        "checkpointer": "postgres" if settings.database_url else "sqlite",
        "cache_enabled": settings.cache_enabled,
        "retrieval_cache_ttl_seconds": settings.retrieval_cache_ttl_seconds,
        "response_cache_enabled": settings.response_cache_enabled,
        "response_cache_ttl_seconds": settings.response_cache_ttl_seconds,
        "cache_policy_version": settings.cache_policy_version,
        "recursion_limit": settings.recursion_limit,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "llm_max_retries": settings.llm_max_retries,
        "mcp_write_timeout_seconds": settings.mcp_write_timeout_seconds,
        "max_routing_visits": settings.run_policy.max_routing_visits,
        "max_plan_items": settings.run_policy.max_plan_items,
        "supervisor_finding_snippet_chars": router._FINDING_SNIPPET_CHARS,
        "max_tool_iterations": settings.run_policy.max_tool_iterations,
        "max_insights": settings.run_policy.max_insights,
        "max_review_rounds": settings.run_policy.max_review_rounds,
        "max_self_check_redrafts": settings.run_policy.max_self_check_redrafts,
        "max_run_tokens": settings.run_policy.max_run_tokens,
        "max_reviewer_notes": settings.run_policy.max_reviewer_notes,
        "memory_store": "postgres" if settings.database_url else "sqlite",
        "max_objective_length": input_validation._MAX_OBJECTIVE_LENGTH,
        "unverified_mark": output_filters.UNVERIFIED_MARK,
        "insight_max_chars": output_filters.INSIGHT_MAX_CHARS,
        "patterns": {
            name: _digest([(label, regex.pattern) for label, regex in getattr(patterns, name)])
            for name in (
                "INJECTION_PATTERNS",
                "EXFILTRATION_PATTERNS",
                "PII_PATTERNS",
                "SECRET_PATTERNS",
            )
        },
    }


def _knowledge() -> dict[str, Any]:
    from market_research_team.caching.store import current_vectorstore_version

    return {
        **_knowledge_static(),
        "index_version": current_vectorstore_version(settings.vectorstore_dir),
    }


def build_manifest() -> dict[str, Any]:
    """Built fresh on every call, deliberately not cached: evals switch
    `settings.llm_provider` mid-process (`--compare`) and a reseed rewrites the
    index stamp under a running app. Hashing a few KB of JSON is sub-millisecond."""

    components = {
        "instructions": _instructions(),
        "model": _model(),
        "tools": _tools(),
        "knowledge": _knowledge(),
        "memory_safety": _memory_safety(),
    }
    component_hashes = {name: _digest(value) for name, value in components.items()}
    return {
        "release": release(),
        "fingerprint": _digest(component_hashes),
        "component_hashes": component_hashes,
        "components": components,
    }


def agent_version() -> str:
    manifest = build_manifest()
    return f"{manifest['release']}+{manifest['fingerprint']}"


def instructions_fingerprint() -> str:
    """Hash of the prompts plus model -- what an LLM-produced cached value depends on."""

    return _digest([_instructions(), _model()])


def trace_metadata() -> dict[str, str]:
    """Flat string metadata for a Langfuse trace (it only accepts string values)."""

    manifest = build_manifest()
    hashes = manifest["component_hashes"]
    model = manifest["components"]["model"]
    return {
        "agent_version": f"{manifest['release']}+{manifest['fingerprint']}",
        "agent_fingerprint": manifest["fingerprint"],
        "git_sha": git_sha(),
        "model": f"{model['provider']}:{model['model']}",
        **{f"{name}_hash": value for name, value in hashes.items()},
    }


def clear_cache() -> None:
    """Drop the cached git SHA -- for tests that set `GIT_SHA`/`GITHUB_SHA`/`CI_COMMIT_SHA`."""

    git_sha.cache_clear()


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Readable lines naming each component whose hash changed, then the keys
    inside it that changed. Tolerates an `old` manifest missing components
    (e.g. one saved before a component was added)."""

    lines: list[str] = []
    old_hashes = old.get("component_hashes", {})
    old_components = old.get("components", {})
    for name, value in new["component_hashes"].items():
        before = old_hashes.get(name)
        if before == value:
            continue
        lines.append(f"{name}: {before} -> {value}")
        old_part = old_components.get(name, {})
        for key, new_value in new["components"][name].items():
            if old_part.get(key) != new_value:
                lines.append(f"    {key}: {old_part.get(key)!r} -> {new_value!r}")
    return lines
