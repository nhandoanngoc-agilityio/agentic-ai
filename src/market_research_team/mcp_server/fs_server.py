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

# Tool-layer guardrails: a report is a few pages of markdown, so anything
# larger is a runaway model or a misuse of the tool, not a report.
_MAX_FILENAME_LENGTH = 128
_MAX_REPORT_BYTES = 256_000


def _resolve_report_path(filename: str) -> Path:
    """Resolve `filename` to a path inside the reports directory.

    Rejects anything that isn't a plain `.md` filename — no directory
    components, no `..` traversal — since this server's only job is
    writing markdown reports, not arbitrary filesystem access.
    """

    if len(filename) > _MAX_FILENAME_LENGTH:
        raise ValueError(f"Report filenames must be at most {_MAX_FILENAME_LENGTH} characters.")

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
def write_report(filename: str, content: str, overwrite: bool = False) -> str:
    """Write markdown content to a file in the reports directory.

    `filename` must be a plain filename ending in `.md`, at most 128
    characters, with no directory components; anything else raises an error
    and nothing is written. `content` is limited to 256,000 bytes (UTF-8).
    An existing report with different content is never replaced unless
    `overwrite` is true (an approved report must not disappear silently);
    writing identical content again succeeds. Returns the absolute path.
    """

    path = _resolve_report_path(filename)
    size = len(content.encode("utf-8"))
    if size > _MAX_REPORT_BYTES:
        raise ValueError(f"Report content is {size} bytes; the limit is {_MAX_REPORT_BYTES}.")
    if path.exists() and not overwrite and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"'{filename}' already exists with different content; nothing written.")
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
