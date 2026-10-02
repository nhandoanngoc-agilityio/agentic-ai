"""Cold-vs-warm runs are scoped separately so the gate reads only the warm run."""

from pathlib import Path

from market_research_team.config import settings
from market_research_team.evaluation import cache_regression
from market_research_team.evaluation.results import EvalResult


def test_cold_and_warm_runs_use_separate_scopes_and_only_warm_is_judged(
    tmp_path: Path, monkeypatch
):
    calls: list[dict] = []

    def fake_run_all(provider=None, *, repeat=0, judge_llm=None):
        calls.append({"audit": settings.audit_log_path, "judge_llm": judge_llm, "repeat": repeat})
        return [EvalResult("reporting", "r", True, "d", "p")]

    monkeypatch.setattr(cache_regression, "run_all", fake_run_all)
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")

    report = cache_regression.run_cold_vs_warm("anthropic", judge_llm="J", run_dir=tmp_path / "p")

    assert calls[0]["audit"] == tmp_path / "p" / "cold" / "audit.jsonl"
    assert calls[1]["audit"] == tmp_path / "p" / "warm" / "audit.jsonl"
    assert calls[0]["judge_llm"] is None and calls[1]["judge_llm"] == "J"
    assert report.warm_audit_path == tmp_path / "p" / "warm" / "audit.jsonl"
