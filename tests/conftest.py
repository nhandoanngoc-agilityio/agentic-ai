import os
from pathlib import Path

import pytest

from market_research_team.config import settings

# Tracing off at import time, not just per test: `graph.py` builds a Langfuse
# client while it is being imported (so `langgraph dev` traces), and test modules
# import it during collection, before any fixture runs. Without this, every pytest
# run sent spans to the real Langfuse project named in `.env` -- the stray
# "Failed to export span batch code: 401" when those keys were wrong.
settings.langfuse_public_key = None
settings.langfuse_secret_key = None
for _name in [name for name in os.environ if name.startswith("LANGFUSE_")]:
    del os.environ[_name]


@pytest.fixture(autouse=True)
def _audit_log_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor writes an audit line at FINISH; keep that out of `data/`."""

    monkeypatch.setattr(settings, "audit_log_path", tmp_path / "audit.jsonl")


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest hermetic even when the local .env has Langfuse keys: with
    tracing on, ratings and harvests would talk to the real Langfuse project.
    Tests that need tracing on set the keys (or patch tracing_enabled) themselves."""

    monkeypatch.setattr(settings, "langfuse_public_key", None)
    monkeypatch.setattr(settings, "langfuse_secret_key", None)


@pytest.fixture(autouse=True)
def _sqlite_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest hermetic even when the local .env sets DATABASE_URL: the
    checkpointer would otherwise reach for Postgres (and time out when it isn't
    running). The live-Postgres tests set it back from the shell themselves."""

    monkeypatch.setattr(settings, "database_url", None)


@pytest.fixture(autouse=True)
def _langfuse_env_restored():
    """`observability._client()` exports settings into `os.environ` with
    `setdefault`; without this, keys set by one test (fake or not) would leak
    into every later test."""

    before = {name: value for name, value in os.environ.items() if name.startswith("LANGFUSE_")}
    yield
    for name in [name for name in os.environ if name.startswith("LANGFUSE_")]:
        if name not in before:
            del os.environ[name]
    os.environ.update(before)
