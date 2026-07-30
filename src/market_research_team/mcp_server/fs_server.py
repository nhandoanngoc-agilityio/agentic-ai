"""Local MCP server exposing filesystem write operations for report output.

Scoped deliberately narrow: the only thing this server can do is write and
list `.md` files inside `settings.reports_dir`. It is not a general-purpose
filesystem tool, so every write is validated against path traversal and
non-report file types before touching disk.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from market_research_team.config import settings

mcp_server = FastMCP("market-research-reports")


def _resolve_report_path(filename: str) -> Path:
    """Resolve `filename` to a path inside the reports directory.

    Rejects anything that isn't a plain `.md` filename — no directory
    components, no `..` traversal — since this server's only job is
    writing markdown reports, not arbitrary filesystem access.
    """

    if not filename.endswith(".md"):
        raise ValueError("Report filenames must end with '.md'.")

    if Path(filename).name != filename:
        raise ValueError(f"'{filename}' must be a plain filename with no directory components.")

    reports_dir = settings.reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidate = (reports_dir / filename).resolve()

    if not candidate.is_relative_to(reports_dir):
        raise ValueError(f"'{filename}' resolves outside the reports directory.")

    return candidate


@mcp_server.tool()
def write_report(filename: str, content: str) -> str:
    """Write markdown content to a file in the reports directory.

    Creates the file if it doesn't exist, or overwrites it if it does.
    Returns the absolute path that was written.
    """

    path = _resolve_report_path(filename)
    path.write_text(content, encoding="utf-8")
    return str(path)


@mcp_server.tool()
def list_reports() -> list[str]:
    """List the filenames of markdown reports currently in the reports directory."""

    reports_dir = settings.reports_dir
    if not reports_dir.exists():
        return []
    return sorted(path.name for path in reports_dir.glob("*.md"))


if __name__ == "__main__":
    mcp_server.run()
