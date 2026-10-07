"""Optional Langfuse tracing, plus local token-usage tracking.

Langfuse tracing is enabled only when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set (see `config.py`); otherwise every Langfuse
helper here is a no-op. This keeps `pytest` hermetic (no network, no
configured client) and keeps unconfigured local runs silent instead of
erroring.

The Langfuse SDK's default client (`get_client()` with no arguments) reads
its credentials from `os.environ`, not from an app's own config object. Since
`config.py` loads `.env` into a pydantic-settings object rather than into
`os.environ`, `_client()` copies the relevant settings into `os.environ`
(only when tracing is enabled) before the first `get_client()` call.

`TokenUsageCallbackHandler` is unrelated to Langfuse: it's a local, always-on
signal (see `security/audit.py`) that exists regardless of whether tracing
is configured, mirroring the local latency/fallback-event tracking.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from market_research_team.config import settings
from market_research_team.security import audit


def tracing_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache(maxsize=1)
def _client():
    from langfuse import get_client

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", settings.langfuse_tracing_environment)
    # Build provenance on every trace; the behavioural version travels as the
    # trace `version` (see `graph.run_graph` and `versioning.py`).
    from market_research_team.versioning import git_sha

    os.environ.setdefault("LANGFUSE_RELEASE", git_sha())

    return get_client()


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """The LangChain/LangGraph callback handler, bound to the configured client.

    Callbacks attached to a graph invocation propagate automatically (via
    contextvars) into every node's internal LLM calls, so nodes don't need to
    accept or forward `RunnableConfig` themselves.
    """
    from langfuse.langchain import CallbackHandler

    _client()  # configure the global client with our credentials first
    return CallbackHandler()


def record_cache_event(layer: str, *, hit: bool, key_hash: str, latency_ms: float) -> None:
    """Emit a cache hit/miss as a point event on the current trace.

    No-op when tracing is disabled, and never raises -- a Langfuse hiccup
    must not break the research pipeline.
    """
    if not tracing_enabled():
        return
    try:
        _client().create_event(
            name=f"cache.{layer}",
            input={"key_hash": key_hash},
            output={"hit": hit},
            metadata={"latency_ms": latency_ms},
        )
    except Exception:
        pass


def record_feedback_score(thread_id: str | None, rating: str, comment: str = "") -> None:
    """Send a thumbs up/down to Langfuse as a `user_feedback` score on the run's
    session (session_id == thread_id). No-op when tracing is off; never
    raises -- a Langfuse hiccup must not break the UI."""

    if not tracing_enabled() or not thread_id:
        return
    try:
        _client().create_score(
            name="user_feedback",
            value=1.0 if rating == "up" else 0.0,
            session_id=thread_id,
            data_type="NUMERIC",
            comment=comment or None,
        )
    except Exception:
        pass


def langfuse_api() -> Any:
    """The Langfuse REST API client (traces, scores, observations)."""

    return _client().api


def flush() -> None:
    """Force-send buffered traces. Call before a short-lived process exits
    (CLI scripts, evals) -- the background flush thread may not get to run."""
    if tracing_enabled():
        _client().flush()


def component_from_tags(tags: list[str] | None) -> str:
    """The call site's component tag (e.g. "supervisor_router").

    Inside a graph run LangGraph puts its own tags first ("seq:step:1"), so
    `tags[0]` attributed every call to that. Framework tags are always
    `namespace:value`; component names never contain a colon.
    """

    return next((tag for tag in tags or [] if ":" not in tag), "untagged")


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Records per-call token usage to the local audit log.

    Bound once per model in `llm.py::get_chat_model()`, so every call site
    gets it automatically. The calling component is read from the run's
    `tags` (each call site passes `config={"tags": [component_name]}`; see
    `component_from_tags`) --
    without a tag, usage is still recorded under `component="untagged"`
    rather than silently dropped.
    """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            component = component_from_tags(tags)
            for generation_list in response.generations:
                for generation in generation_list:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None) if message else None
                    if not usage:
                        continue
                    audit.record(
                        "llm_usage",
                        audit.current_thread_id(),
                        component=component,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )
        except Exception:
            pass
