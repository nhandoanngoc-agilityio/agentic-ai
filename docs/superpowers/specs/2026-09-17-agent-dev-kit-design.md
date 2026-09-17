# Agent Development Kit for this repo — design

**Date:** 2026-09-17
**Goal:** Solo productivity. Make Claude Code work better on this repo by adding
the first four layers of the "Agent Development Kit" (context, skills, hooks,
subagents). Plugins and Agent Teams are out of scope.

## Decisions already made

- Config is committed to git. `.gitignore` stops ignoring `.claude/*`; only
  `.claude/settings.local.json` and `.claude/worktrees/` stay ignored.
  `.teamai/*` and `docs/superpowers/*` ignores are unchanged.
- Existing teamai hooks in `.claude/settings.json` are kept as they are; new
  hooks are added beside them.
- Agents are model-pinned to control cost: cheap models for frequent work,
  Fable only for on-demand review and debugging.

## Layer 1 — Context: `CLAUDE.md`

Root file, target 50–70 lines. Sections:

1. **What this is** — one paragraph: LangGraph supervisor + research /
   analytics / reporting agents, local MCP filesystem server. Link to
   `docs/architecture.md` instead of repeating it.
2. **Commands** — install, `pytest`, `ruff check src tests scripts`,
   `python scripts/seed_vectorstore.py`, `python scripts/run_graph_cli.py "<objective>"`,
   `langgraph dev --no-browser`, `python scripts/run_evals.py`.
3. **Conventions** (derived from the code):
   - Every graph node is registered through `with_error_boundary` in
     `graph.py`; never add a bare node.
   - Nodes take `AgentState` and return a partial update dict.
   - `pytest` is hermetic: fake LLMs, no API keys, no seeded vector store.
   - `ruff` line length 100, rules E/F/I/UP.
   - Design specs go in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`.
   - Config lives in `config.py` (pydantic-settings); read `.env.example`, never `.env`.
4. **Cost rules** — `run_evals.py` and `run_graph_cli.py` call real LLM APIs.
   Ask before running them. Use the `explorer` agent for codebase searches and
   `test-runner` for test runs to keep the main context small.
5. **Delegation map** — which agent to use for what (see Layer 4).

Auto memory needs no files; it is on by default and Claude manages it.

## Layer 2 — Skills: `.claude/skills/<name>/SKILL.md`

Each skill has frontmatter `name` and `description` (the description is the
trigger text), a short procedure, and a list of the real files involved.
No `scripts/` or `references/` folders unless a step needs one.

| Skill | Triggers when | Procedure covers |
|---|---|---|
| `langgraph-node` | adding or changing a graph node, agent, or route | new state fields in `state.py`; node module under `agents/<name>/node.py`; register with `with_error_boundary` in `graph.py`; add the route in `supervisor/router.py` and the conditional-edge map; write a hermetic test following `tests/test_supervisor_routing.py` and `tests/test_reporting_node.py` patterns |
| `run-evals` | running or interpreting evals, changing a prompt, switching models | prerequisites (API key, seeded store); `run_evals.py` flags (`--provider`, `--compare`, `--langsmith`); results in `data/eval_results/`; add cases in `evaluation/golden_dataset.py`; cost warning |
| `mcp-tool` | adding a tool to the local MCP server or exposing it to an agent | add the tool in `mcp_server/fs_server.py`; client wiring in `agents/reporting/mcp_client.py`; test via the real subprocess pattern in `tests/test_mcp_server.py` |
| `rag-tuning` | changing chunking, retrieval, or reranking | knobs in `config.py` (`section_chunk_size`, `leaf_chunk_size`, `leaf_chunk_overlap`, `reranker_model_name`); code in `ingestion/chunking.py`, `agents/research/retriever.py`, `agents/research/reranker.py`; reseed after changes; tests `test_chunking.py`, `test_retriever.py`, `test_reranker.py` |

## Layer 3 — Hooks: `.claude/hooks/*.sh` + `settings.json`

Three bash scripts, each reading the tool-call JSON from stdin. `jq` is not
assumed; parsing uses `python3 -c` since Python is guaranteed here.

| Script | Event / matcher | Behaviour |
|---|---|---|
| `ruff-on-edit.sh` | `PostToolUse`, matcher `Edit\|Write\|MultiEdit` | If `tool_input.file_path` ends in `.py` and is inside the repo, run `ruff format <file>` then `ruff check --fix <file>` using `.venv/bin/ruff` when present, else `ruff` on PATH. Never fails the tool call (exit 0). |
| `guard-bash.sh` | `PreToolUse`, matcher `Bash` | Reads `tool_input.command`. Blocks (exit 2 with a one-line reason on stderr) when it matches: `rm -rf` / `rm -fr` targeting `/`, `~`, `.`, `data`, `reports`, `src`, or `.git`; `git push` with `--force` or `-f`; `git clean -f`; any command that writes to or removes `.env`. Everything else exits 0. |
| `protect-secrets.sh` | `PreToolUse`, matcher `Read\|Edit\|Write\|MultiEdit` | Reads `tool_input.file_path`. Blocks (exit 2) when the basename is `.env` or the path ends in `data/checkpoints.sqlite`. `.env.example` is allowed. |

`settings.json` gains a `PreToolUse` array and extends the existing
`PostToolUse` array. The teamai entries are preserved verbatim.

## Layer 4 — Subagents: `.claude/agents/<name>.md`

Frontmatter fields: `name`, `description`, `model`, `tools`. Body is the
system prompt, kept under 30 lines each.

| Agent | Model | Tools | Purpose |
|---|---|---|---|
| `explorer` | haiku | Read, Grep, Glob, Bash (read-only use) | Find where things live and summarise. Returns file paths and short excerpts, never whole files. |
| `test-runner` | sonnet | Bash, Read, Grep | Runs `pytest` (optionally a subset) and `ruff check`. Reports only failures with the assertion and the relevant lines; says "all green" otherwise. Never edits code. |
| `code-reviewer` | fable | Read, Grep, Glob, Bash | Reviews a diff for correctness against the conventions in CLAUDE.md and the security-hardening specs. Findings ranked by severity. Read-only. |
| `langgraph-debugger` | fable | Read, Grep, Glob, Bash | Diagnoses routing loops, state-shape bugs, checkpoint/resume issues. Follows systematic-debugging: reproduce, locate, explain root cause, propose the minimal fix. Does not edit. |

## Git changes

`.gitignore`: replace the line `.claude/*` with

```
.claude/settings.local.json
.claude/worktrees/
```

## Verification

- `python3 -m json.tool .claude/settings.json` succeeds.
- Each hook script is executable and is exercised with sample stdin JSON:
  guard-bash blocks `rm -rf data` and `git push --force`, allows `pytest`;
  protect-secrets blocks `.env` and allows `.env.example`;
  ruff-on-edit reformats a deliberately mis-formatted temp `.py` file.
- Every SKILL.md and agent file has the required frontmatter fields.
- `git status` shows `.claude/` files as trackable after the ignore change.

## Out of scope

Plugin packaging (`.claude-plugin/`), Agent Teams, a global
`~/.claude/CLAUDE.md`, changes to teamai hooks, and any change to the
Python source.
