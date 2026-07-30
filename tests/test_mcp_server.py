"""Tests for the local filesystem-write MCP server.

Uses the MCP SDK's in-process client/server session (no subprocess, no
stdio transport) so these run as fast, hermetic unit tests.
"""

from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from market_research_team.config import settings
from market_research_team.mcp_server.fs_server import mcp_server


@pytest.fixture(autouse=True)
def _use_tmp_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")


async def test_list_tools_exposes_write_and_list() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        tools = await session.list_tools()

    tool_names = {tool.name for tool in tools.tools}
    assert tool_names == {"write_report", "list_reports"}


async def test_write_report_creates_file_with_content() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        result = await session.call_tool(
            "write_report", {"filename": "acme.md", "content": "# Acme\n\nSummary."}
        )

    assert result.isError is False
    written_path = settings.reports_dir / "acme.md"
    assert written_path.read_text(encoding="utf-8") == "# Acme\n\nSummary."
    assert result.content[0].text == str(written_path)


async def test_write_report_overwrites_existing_file() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        await session.call_tool("write_report", {"filename": "x.md", "content": "first"})
        await session.call_tool("write_report", {"filename": "x.md", "content": "second"})

    assert (settings.reports_dir / "x.md").read_text(encoding="utf-8") == "second"


async def test_write_report_rejects_non_markdown_filename() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        result = await session.call_tool("write_report", {"filename": "x.txt", "content": "hi"})

    assert result.isError is True
    assert not (settings.reports_dir / "x.txt").exists()


async def test_write_report_rejects_path_traversal() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        result = await session.call_tool(
            "write_report", {"filename": "../escape.md", "content": "hi"}
        )

    assert result.isError is True
    assert not (settings.reports_dir.parent / "escape.md").exists()


async def test_write_report_rejects_nested_path() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        result = await session.call_tool(
            "write_report", {"filename": "sub/dir.md", "content": "hi"}
        )

    assert result.isError is True
    assert not (settings.reports_dir / "sub").exists()


async def test_list_reports_returns_written_filenames() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        await session.call_tool("write_report", {"filename": "b.md", "content": "b"})
        await session.call_tool("write_report", {"filename": "a.md", "content": "a"})
        result = await session.call_tool("list_reports", {})

    assert result.isError is False
    assert result.structuredContent == {"result": ["a.md", "b.md"]}


async def test_list_reports_empty_when_reports_dir_missing() -> None:
    async with create_connected_server_and_client_session(mcp_server) as session:
        result = await session.call_tool("list_reports", {})

    assert result.isError is False
    assert result.structuredContent == {"result": []}
