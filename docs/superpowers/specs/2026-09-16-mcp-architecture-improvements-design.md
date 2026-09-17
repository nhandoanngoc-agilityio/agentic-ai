# MCP Architecture Improvements — Design Spec

**Date:** 2026-09-16
**Status:** Approved for planning
**Sub-project of:** repo-wide practice improvements (MCP architecture is sub-project 3 of 6; sub-project 1, LangSmith eval upgrade, and sub-project 2, human-in-the-loop approval, are done and merged to `main`. Sub-project 4, the streaming comparison UI, is done with an open MR awaiting merge. Sub-projects 5/6, security and best-practices, are folded into whichever sub-project touches the relevant code, per the standing decision recorded in sub-project 1's spec — not standalone here either.)

## Context

The only MCP server in this repo, `src/market_research_team/mcp_server/fs_server.py`, is deliberately narrow: two tools, `write_report(filename, content)` and `list_reports()`, both scoped to `settings.reports_dir` with real path-traversal/extension validation. It runs exclusively over stdio, spawned as a trusted local subprocess by `agents/reporting/mcp_client.py`, which is its only consumer (the Reporting Agent node). No other MCP server or client exists anywhere in this repo.

Three concrete gaps in that narrow scope, not a request to broaden what the server covers:

1. There is no way to read a report's content back through MCP — `list_reports()` only returns filenames. Verifying MCP-written output today means reading the filesystem directly (in tests, and in any hypothetical external tooling), bypassing MCP entirely.
2. The report-drafting system prompt is hardcoded as a private constant inside `agents/reporting/node.py`'s `draft_report()` — not reusable by any other MCP client.
3. The server only runs over stdio for a single trusted internal caller. There's no way for another client (a different process, a different tool, a human inspecting reports from outside this codebase) to connect to it at all, let alone securely.

## Scope

**In scope:** MCP `resources` and `prompts` primitives added to the existing server (still report-scoped, not broadened to other agent domains); a second network entrypoint (`streamable-http` transport) alongside the unchanged stdio one; self-issued, HMAC-signed, per-client-scoped bearer-token auth for that network entrypoint.

**Explicitly out of scope:**
- Broadening the MCP server to cover Research or Analytics agent capabilities — it stays reports-only, per explicit user decision.
- Changing `reporting_node.py`/`draft_report()` to fetch its own system prompt from the new MCP prompt at draft time. The prompt is published via MCP for *external* reuse; the Reporting Agent's internal drafting path keeps using its own hardcoded string, unchanged. Introducing an async MCP round-trip into the already-working, already-tested draft/redraft interrupt loop for the sake of removing one string duplication was assessed as not worth the added risk, and the user confirmed keeping this conservative split.
- Integrating with an external OAuth/OIDC identity provider (Auth0, Keycloak, etc.). Tokens are self-issued and self-verified by this server using a shared HMAC secret — real per-client scoped auth, without an external operational dependency inappropriate for a local training repo.
- A persistent client registry/database. Access control is "whoever holds `MCP_JWT_SECRET` can mint a token naming any client id and scopes" — there is no separate store of "registered" clients to manage or query.
- Any change to the stdio path's behavior, the two existing tools' behavior, or `mcp_client.py`'s use of them. The Reporting Agent's existing flow is unaffected by any of this.

## Goals

1. Add a `report://{filename}` resource returning a report's markdown content, so reading a report no longer requires touching the filesystem directly — usable by tests, external tooling, or any MCP client.
2. Add a `report_draft_template` prompt exposing the report-drafting system prompt as a reusable MCP prompt, extracted from (not removed from) `reporting/node.py`'s private constant.
3. Add a `streamable-http` entrypoint (`mcp_server/http_server.py`) serving the *same* tool/resource/prompt definitions as the stdio server, so nothing is duplicated between the two transports.
4. Gate the HTTP entrypoint with self-issued, HMAC-signed JWT bearer tokens carrying a client identity (`sub`) and scopes (`reports:read` for the resource and `list_reports`, `reports:write` for `write_report`), verified on every request.
5. Provide a script to mint a token for a named client with named scopes, backed by a shared secret read from `MCP_JWT_SECRET`.

## Non-goals

See Scope's "Explicitly out of scope" — restated here as the binding non-goal list this spec's plan must honor: no broadening beyond reports, no change to `draft_report()`'s prompt sourcing, no external IdP integration, no persistent client registry, no change to the stdio path or the two existing tools' behavior.

## Architecture

```
src/market_research_team/mcp_server/
├── fs_server.py          # unchanged tools (write_report, list_reports)
│                          #   + new: report://{filename} resource
│                          #   + new: report_draft_template prompt
│                          # exports `mcp_server: FastMCP` (unchanged export)
│                          # `if __name__ == "__main__": mcp_server.run()` (unchanged: stdio)
│
├── auth.py                # new: TokenVerifier implementation (HMAC-signed JWT:
│                          #   signature + expiry + scope check), reading
│                          #   MCP_JWT_SECRET from the environment
│
└── http_server.py         # new: builds a Starlette app wrapping
                           #   mcp_server.streamable_http_app() with bearer-auth
                           #   middleware using auth.py's TokenVerifier;
                           #   `if __name__ == "__main__": uvicorn.run(app, ...)`

scripts/
└── issue_mcp_token.py     # new: CLI — mint an HMAC-signed JWT for a given
                           #   client id + space-separated scopes, using
                           #   MCP_JWT_SECRET
```

`mcp_client.py` (the Reporting Agent's client) is untouched — it still spawns `fs_server.py` over stdio exactly as today, never touches `http_server.py` or auth at all. The new resource and prompt are registered on the same `mcp_server` instance `mcp_client.py` already talks to, so they're automatically available over stdio too (no reason to hide them from the internal client) — but nothing in this repo's graph code consumes them; they exist for external MCP clients.

## Components

### `fs_server.py` — new resource

```python
@mcp_server.resource("report://{filename}")
def read_report(filename: str) -> str:
    """Return a written report's markdown content, or raise if it doesn't exist /
    fails the same filename validation write_report already enforces."""
```

Reuses `_resolve_report_path(filename)` (existing private helper — same path-traversal/extension
validation as `write_report`, not a separate/weaker check) and reads the resolved path's text,
raising if it doesn't exist.

### `fs_server.py` — new prompt

```python
@mcp_server.prompt()
def report_draft_template(objective: str, findings_summary: str, results_summary: str) -> str:
    """The report-drafting system prompt, extracted (not removed) from
    agents/reporting/node.py's private _SYSTEM_PROMPT + human-content-building logic,
    published here for reuse by other MCP clients. reporting/node.py's own drafting
    path is unaffected -- see Non-goals."""
```

### `auth.py` — token verification

A `TokenVerifier` (matching whatever protocol the installed `mcp` SDK's auth plumbing expects —
confirmed to exist via the SDK's `BearerAuthBackend(token_verifier)` wiring; exact interface
shape to be pinned during implementation planning against the actually-installed SDK version, the
same way this session has verified every other library integration against real installed code
rather than assumed signatures) that:
- Decodes and verifies an HMAC-SHA256-signed JWT against `MCP_JWT_SECRET`.
- Rejects expired or malformed tokens.
- Extracts `sub` (client id, for logging/audit) and `scope` (space-separated scopes) claims.
- Exposes what's needed for `RequireAuthMiddleware` to enforce required scopes per route (`reports:read` for the resource and `list_reports`, `reports:write` for `write_report`).

### `http_server.py` — network entrypoint

Builds a Starlette app: mounts `mcp_server.streamable_http_app()` behind the bearer-auth
middleware from `auth.py`, matching the SDK's documented `AuthenticationMiddleware` +
`RequireAuthMiddleware` wiring pattern. Run via `uvicorn market_research_team.mcp_server.http_server:app --port <port>` (port to be fixed to a specific default during planning, distinct from every other port already documented in this repo — 2024/2025 for the two `langgraph dev` deployments, 3000 for the Next.js UI).

### `scripts/issue_mcp_token.py`

```
python scripts/issue_mcp_token.py <client_id> --scopes reports:read reports:write [--ttl-seconds N]
```

Prints a signed JWT to stdout. Reads `MCP_JWT_SECRET` from the environment (fails fast with a
clear message if unset, matching `scripts/setup_env.py`'s existing fail-fast philosophy). No
persistent state written anywhere — see Non-goals on the absence of a client registry.

## Data flow

**Stdio (unchanged):** `mcp_client.py` spawns `fs_server.py` as today; the Reporting Agent calls
`write_report` at the end of a drafting round. Nothing about this path changes.

**HTTP (new):** An operator runs `python scripts/issue_mcp_token.py external-tool --scopes reports:read` once to get a token for a given external client, then that client calls the
`streamable-http` endpoint with `Authorization: Bearer <token>`. `http_server.py`'s middleware
verifies the token via `auth.py`, checks the route's required scope, and — only if both pass —
forwards the request into the same `mcp_server` instance's resource/tool handlers.

## Error handling

- Invalid/missing bearer token → 401, matching the SDK's standard `WWW-Authenticate` response
  shape (no custom error format invented).
- Valid token, insufficient scope → 403.
- `read_report` on a nonexistent or invalid filename → same validation errors `write_report`
  already raises for bad filenames (reused helper), plus a clear "not found" for a valid-but-
  nonexistent filename.
- `MCP_JWT_SECRET` unset when starting `http_server.py` or running `issue_mcp_token.py` → fail
  fast with a clear message before doing anything else, not a cryptic downstream error.

## Testing

- `tests/test_mcp_server.py`: new hermetic cases for `report://{filename}` (via
  `session.read_resource(...)`, matching this file's existing
  `create_connected_server_and_client_session` pattern) and `report_draft_template` (via
  `session.get_prompt(...)`) — both in-process, no subprocess, no network, consistent with every
  existing test in this file.
- New `tests/test_mcp_http_server.py`: `TestClient`-based, matching the rigor already established
  for `comparison_api.py`'s CORS tests in an earlier sub-project — no token → 401; valid token,
  wrong scope → 403; valid token, correct scope → success, reusing the same resource/tool
  handlers exercised in the stdio tests.
- New `tests/test_issue_mcp_token.py`: mint → verify round-trip, expired token rejected, tampered
  signature rejected, missing `MCP_JWT_SECRET` fails fast with a clear message.

## Documentation updates

- README: document the HTTP entrypoint and token-minting script as an optional, separate path
  from the graph's normal operation — nothing about the existing "Getting started" flow changes,
  since the Reporting Agent never uses this path.
- `fs_server.py`'s module docstring: update to describe the resource/prompt additions alongside
  the existing tool-scoping description.

## Success criteria

1. `pytest -q` and `ruff check src tests scripts` stay clean, with the new test files' cases
   passing alongside the full existing suite.
2. `session.read_resource("report://<name>.md")` returns a previously-written report's exact
   content over the existing hermetic in-process test pattern.
3. A request to the HTTP entrypoint with no token gets 401; with a token scoped only
   `reports:read` calling `write_report` gets 403; with a token scoped `reports:write` calling
   `write_report` succeeds and the file lands exactly where the stdio path would put it.
4. `scripts/run_graph_cli.py`'s existing end-to-end flow (stdio path) is unaffected — confirmed by
   the existing `test_reporting_pipeline.py`/`test_mcp_server.py` cases continuing to pass
   unmodified.
