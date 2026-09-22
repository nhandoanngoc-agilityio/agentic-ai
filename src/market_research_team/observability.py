"""Optional Langfuse tracing.

Tracing is enabled only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
are set (see `config.py`); otherwise every helper here is a no-op. This keeps
`pytest` hermetic (no network, no configured client) and keeps unconfigured
local runs silent instead of erroring.

The Langfuse SDK's default client (`get_client()` with no arguments) reads
its credentials from `os.environ`, not from an app's own config object. Since
`config.py` loads `.env` into a pydantic-settings object rather than into
`os.environ`, `_client()` copies the relevant settings into `os.environ`
(only when tracing is enabled) before the first `get_client()` call.
"""

from __future__ import annotations

import os
from functools import lru_cache

from market_research_team.config import settings


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


def flush() -> None:
    """Force-send buffered traces. Call before a short-lived process exits
    (CLI scripts, evals) -- the background flush thread may not get to run."""
    if tracing_enabled():
        _client().flush()
