"""harvest_failures.py and promote_case.py wiring (hermetic)."""

import importlib.util
import json
from pathlib import Path

import pytest

from market_research_team.config import settings


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harvest_cli = _load("harvest_failures")
promote_cli = _load("promote_case")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    audit = tmp_path / "audit.jsonl"
    event = {
        "ts": "2026-09-29T10:00:00+00:00",
        "event": "user_satisfaction",
        "thread_id": "t1",
        "objective": "Compare Acme pricing",
        "rating": "down",
    }
    audit.write_text(json.dumps(event) + "\n")
    monkeypatch.setattr(settings, "audit_log_path", audit)
    monkeypatch.setattr(settings, "candidates_dir", tmp_path / "q")
    monkeypatch.setattr(settings, "regressions_path", tmp_path / "regressions.jsonl")
    return tmp_path


def test_parse_since():
    assert harvest_cli.parse_since("7d") is not None
    assert harvest_cli.parse_since("24h") is not None
    assert harvest_cli.parse_since("2026-09-01").year == 2026
    with pytest.raises(ValueError):
        harvest_cli.parse_since("soon")


def test_harvest_then_promote_flow(env: Path, capsys):
    assert harvest_cli.main(["--since", "2026-01-01", "--no-langfuse"]) == 0
    cid = next((env / "q" / "pending").glob("*.json")).stem

    assert promote_cli.main(["promote", cid]) == 2  # no expectations yet
    assert "at least one expectation" in capsys.readouterr().err

    path = env / "q" / "pending" / f"{cid}.json"
    data = json.loads(path.read_text())
    data["expectations"]["required_facts"] = ["$49"]
    path.write_text(json.dumps(data))

    assert promote_cli.main(["promote", cid]) == 0
    assert json.loads((env / "regressions.jsonl").read_text())["id"] == cid


def test_list_show_reject_and_unknown_id(env: Path, capsys):
    harvest_cli.main(["--since", "2026-01-01", "--no-langfuse"])
    cid = next((env / "q" / "pending").glob("*.json")).stem

    assert promote_cli.main(["list"]) == 0
    assert cid in capsys.readouterr().out
    assert promote_cli.main(["show", cid]) == 0
    assert promote_cli.main(["reject", cid, "--reason", "noise"]) == 0
    assert promote_cli.main(["show", cid]) == 2


prune_cli = _load("prune_data")


def test_prune_cli_is_a_dry_run_by_default(env: Path, monkeypatch, capsys):
    from market_research_team.feedback.retention import PruneReport

    calls: list = []

    def fake_prune(cutoff, *, apply):
        calls.append(apply)
        return PruneReport(applied=apply, audit_removed=3)

    monkeypatch.setattr(prune_cli, "prune", fake_prune)

    assert prune_cli.main(["--days", "30"]) == 0
    assert calls == [False] and "DRY RUN" in capsys.readouterr().out
    assert prune_cli.main(["--apply"]) == 0
    assert calls == [False, True]
