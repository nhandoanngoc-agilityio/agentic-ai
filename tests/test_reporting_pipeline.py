"""Integration test for the Reporting Agent's MCP client wiring.

Spawns the real local MCP filesystem server as a subprocess (same as
production), so this is slower than the rest of the suite but is the test
that actually proves Day 9's deliverable: the Reporting Agent can reach
the MCP server over stdio and produce a real file on disk — and that the
subprocess honors an overridden `settings.reports_dir` rather than
silently writing into the real project directory (a bug caught and fixed
while building this).
"""

from pathlib import Path

import pytest

from market_research_team.agents.reporting.node import run_reporting_pipeline
from market_research_team.config import settings

_FINDING = {
    "source": "competitor_acme.md",
    "content": "Acme prices at $49/seat.",
    "relevance_score": 0.9,
}
_RESULT = {"metric": "mean", "value": 49.0, "detail": "mean([49]) = 49.0"}


@pytest.fixture(autouse=True)
def _use_tmp_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")


async def test_run_reporting_pipeline_writes_a_real_file_via_mcp() -> None:
    report_path = await run_reporting_pipeline(
        "Assess Acme vs Globex pricing strategy", [_FINDING], [_RESULT]
    )

    written = Path(report_path)
    assert written.exists()
    assert written.parent == settings.reports_dir
    content = written.read_text(encoding="utf-8")
    assert "Assess Acme vs Globex pricing strategy" in content
    assert "Acme prices at $49/seat." in content


async def test_run_reporting_pipeline_does_not_touch_the_real_reports_dir() -> None:
    project_reports_dir = Path(__file__).resolve().parents[1] / "reports"
    before = set(project_reports_dir.glob("*.md"))

    await run_reporting_pipeline("Assess pricing", [_FINDING], [])

    after = set(project_reports_dir.glob("*.md"))
    assert before == after
