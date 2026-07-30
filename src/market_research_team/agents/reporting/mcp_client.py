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


async def load_reporting_tools() -> list[BaseTool]:
    """Spawn the local MCP filesystem server and load its tools as LangChain tools.

    The server is a separate process, so it doesn't share this process's
    `settings` object — its `REPORTS_DIR` is passed explicitly via the
    subprocess environment rather than relying on both processes
    independently reading the same `.env` file, which would silently break
    if this process's `settings.reports_dir` is overridden at runtime
    (e.g. in tests).
    """

    connections = {
        "market_research_reports": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "market_research_team.mcp_server.fs_server"],
            "env": {**os.environ, "REPORTS_DIR": str(settings.reports_dir)},
        }
    }
    client = MultiServerMCPClient(connections)
    return await client.get_tools()
