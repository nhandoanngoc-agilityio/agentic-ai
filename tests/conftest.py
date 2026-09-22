from pathlib import Path

import pytest

from market_research_team.config import settings


@pytest.fixture(autouse=True)
def _audit_log_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor writes an audit line at FINISH; keep that out of `data/`."""

    monkeypatch.setattr(settings, "audit_log_path", tmp_path / "audit.jsonl")
