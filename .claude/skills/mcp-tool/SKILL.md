---
name: mcp-tool
description: Use when adding a tool to the local MCP filesystem server or exposing an MCP tool to an agent node. Covers server definition, client wiring, path safety, and the in-process test pattern.
---

# Adding an MCP tool

1. **Define it** in `src/market_research_team/mcp_server/fs_server.py` next to `write_report` and `list_reports`:
   ```python
   @mcp_server.tool()
   def my_tool(arg: str) -> str:
       """One-line description shown to the LLM."""
   ```
   Any path argument must go through `_resolve_report_path` (rejects traversal and nested paths and forces `.md`). Never build paths from raw input.
2. **Expose it.** `load_reporting_tools()` in `src/market_research_team/agents/reporting/mcp_client.py` loads every tool the server exposes, so no client change is needed unless a different agent should get it.
3. **Use it** in the node: bind the tools from `load_reporting_tools()` to the model and handle the tool result like `agents/reporting/node.py` does.
4. **Test** in `tests/mcp/test_mcp_server.py` using the SDK's in-process client/server session (no subprocess). Add: one happy-path test, one that rejects a bad path, and one for the empty/missing case.
5. Run the `test-runner` agent.
