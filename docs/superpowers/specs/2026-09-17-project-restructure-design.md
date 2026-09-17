# Project folder restructure — design

**Date:** 2026-09-17
**Goal:** Rearrange the repo to follow the "production-ai-app" reference layout where this
project has real code to put there. Pure moves and renames; no behaviour change.
**Branch:** `agent-dev-kit-and-restructure` (continues the Claude agent-dev-kit work).

## Decisions

- Only folders that hold real code are created. No `observability/`, `services/`,
  `prompts/`, Docker files, or empty scaffolding.
- Prompts stay hardcoded in their nodes (a later change, tracked by the MCP
  architecture spec).
- Two in-flight worktrees (`security-hardening-prompt-injection`,
  `langsmith-eval-upgrade`) are left alone; the user rebases them later.
- Historical specs and plans under `docs/superpowers/` keep old paths.

## Package moves (`src/market_research_team/`)

| From | To |
|---|---|
| `agents/research/retriever.py` | `retrieval/retriever.py` |
| `agents/research/reranker.py` | `retrieval/reranker.py` |
| `agents/research/query_rewriter.py` | `retrieval/query_rewriter.py` |
| `validation.py` | `security/input_validation.py` |
| `supervisor/router.py` | `agents/supervisor/router.py` |

`agents/research/node.py` stays. `supervisor/__init__.py` moves with the package.
New packages `retrieval/` and `security/` get an empty `__init__.py`.
Import rewrite rules (applied to `src/`, `tests/`, `scripts/`):

| Old module path | New module path |
|---|---|
| `market_research_team.agents.research.retriever` | `market_research_team.retrieval.retriever` |
| `market_research_team.agents.research.reranker` | `market_research_team.retrieval.reranker` |
| `market_research_team.agents.research.query_rewriter` | `market_research_team.retrieval.query_rewriter` |
| `market_research_team.validation` | `market_research_team.security.input_validation` |
| `market_research_team.supervisor` | `market_research_team.agents.supervisor` |

`from market_research_team.agents.research import retriever` style imports are rewritten
by hand where the scripted rule cannot express them.

## Test grouping (`tests/`)

| Folder | Files |
|---|---|
| `tests/agents/` | test_analytics_loop, test_analytics_tools, test_reporting_node, test_reporting_node_interrupt, test_reporting_pipeline, test_supervisor_decision, test_supervisor_routing |
| `tests/retrieval/` | test_query_rewriter, test_reranker, test_retriever |
| `tests/ingestion/` | test_chunking, test_ingestion_loaders |
| `tests/evaluation/` | test_eval_checks, test_eval_golden_dataset, test_eval_offline, test_langsmith_eval |
| `tests/graph/` | test_graph_discard_terminates, test_guardrails, test_run_graph_safety, test_state, test_llm |
| `tests/checkpointing/` | test_checkpointing, test_checkpointing_postgres |
| `tests/mcp/` | test_mcp_server |
| `tests/security/` | test_validation |
| `tests/scripts/` | test_run_evals_cli, test_run_graph_cli_validation, test_setup_env |

No `__init__.py` in test folders; basenames are unique so pytest's default import
mode works. `pyproject.toml` `testpaths` is unchanged.

## Frontend

`ui/` → `frontend/` via `git mv`. Untracked build output (`.next/`, `node_modules/`)
moves with the directory on disk. README references updated.

## Root

`AGENTS.md` created: two lines pointing tools other than Claude Code at `CLAUDE.md`.

## Docs and Claude config updates

README project-layout tree and `ui/` mentions; `CLAUDE.md`; `.claude/agents/explorer.md`;
`.claude/agents/langgraph-debugger.md`; `.claude/skills/rag-tuning/SKILL.md`;
`.claude/skills/langgraph-node/SKILL.md`; `docs/architecture.md` gets a dated entry.

## Verification

- `grep -rn` for each old module path in `src tests scripts` returns nothing.
- `pytest -q`: same count as before the move (157 passed, 1 skipped).
- `ruff check src tests scripts` clean (import sorting will be re-run by `ruff check --fix`).
- `python -c "from market_research_team.graph import graph"` succeeds.
- `cd frontend && npm test` passes.
