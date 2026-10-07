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


@pytest.fixture(autouse=True)
def _offline_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test that builds the real graph runs the planner node; keep it off
    the network with the deterministic one-item plan (the planner's own
    fallback). Planner tests monkeypatch `run_planner` back themselves."""

    from market_research_team.agents.planner import node as planner_node_module

    monkeypatch.setattr(planner_node_module, "run_planner", planner_node_module.fallback_plan)


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy provider keys for every test, real ones hidden.

    `ChatOpenAI` refuses to construct without a key, so a test that only builds
    a client passed on a machine whose shell exports OPENAI_API_KEY and failed
    in CI, which has none. The dummies make that hermetic, and clearing the
    settings keys (read from a local .env) means no test can see a real key.
    Tests that exercise key handling set their own values.
    """

    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
