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

from market_research_team.agents.reporting.node import (
    draft_report,
    report_filename,
    write_report_via_mcp,
)
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


async def _draft_and_write(objective: str, findings: list, results: list) -> str:
    """Draft (deterministic fallback: no model) and write through the real
    MCP server -- the review node's write path, minus the approval step."""

    content = draft_report(objective, findings, results, None)  # type: ignore[arg-type]
    return await write_report_via_mcp(report_filename(objective, None), content)


async def test_a_report_is_written_to_disk_through_the_real_mcp_server() -> None:
    report_path = await _draft_and_write(
        "Assess Acme vs Globex pricing strategy", [_FINDING], [_RESULT]
    )

    written = Path(report_path)
    assert written.exists()
    assert written.parent == settings.reports_dir
    content = written.read_text(encoding="utf-8")
    assert "Assess Acme vs Globex pricing strategy" in content
    assert "49" in content


async def test_the_server_honours_an_overridden_reports_dir() -> None:
    project_reports_dir = Path(__file__).resolve().parents[2] / "reports"
    before = set(project_reports_dir.glob("*.md"))

    await _draft_and_write("Assess pricing", [_FINDING], [])

    after = set(project_reports_dir.glob("*.md"))
    assert before == after


def test_the_server_gets_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_research_team.agents.reporting.mcp_client import server_environment

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host/db")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-should-not-leak")

    env = server_environment()

    assert env["REPORTS_DIR"] == str(settings.reports_dir)
    assert "PATH" in env
    assert not {"OPENAI_API_KEY", "DATABASE_URL", "LANGFUSE_SECRET_KEY"} & set(env)
