from pathlib import Path

import pytest

from market_research_team.config import settings


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
