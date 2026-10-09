"""MCP client wiring: spawns the local filesystem-write MCP server over stdio
and loads its tools as LangChain tools.

Kept as a separate module from `mcp_server/fs_server.py` on purpose: the
server (Day 8) and the client that consumes it (Day 9) are genuinely
different processes here, not just different files.
"""

import os
import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from market_research_team.config import settings

# Inherited by the server process; everything else (provider API keys,
# DATABASE_URL, Langfuse keys) stays out. The server only writes files.
_PASSTHROUGH_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "VIRTUAL_ENV",
    "PYTHONPATH",
)


def server_environment() -> dict[str, str]:
    """The MCP server's environment: what a Python process needs, plus REPORTS_DIR."""

    env = {name: os.environ[name] for name in _PASSTHROUGH_ENV if name in os.environ}
    env["REPORTS_DIR"] = str(settings.reports_dir)
    return env


async def load_reporting_tools() -> list[BaseTool]:
    """Spawn the local MCP filesystem server and load its tools as LangChain tools.

    The server is a separate process, so it doesn't share this process's
    `settings` object — its `REPORTS_DIR` is passed explicitly via the
    subprocess environment rather than relying on both processes
    independently reading the same `.env` file, which would silently break
    if this process's `settings.reports_dir` is overridden at runtime
    (e.g. in tests).
    """

    # Least privilege: an allowlisted environment, and the reports directory
    # as the working directory -- the server's settings read `.env` from the
    # working directory, so the project root would hand it the API keys anyway.
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    connections = {
        "market_research_reports": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "market_research_team.mcp_server.fs_server"],
            "env": server_environment(),
            "cwd": str(settings.reports_dir),
        }
    }
    client = MultiServerMCPClient(connections)
    return await client.get_tools()
