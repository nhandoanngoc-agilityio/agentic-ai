# Architecture Notes

Living design log for the Market & Competitor Research Analyst Team. Update
this alongside each day's work rather than after the fact.

## Day 1 — Scaffolding

- Package layout finalized under `src/market_research_team/` (see root
  [README.md](../README.md) for the full tree and rationale).
- Default provider choices: `langchain-anthropic` for the LLM,
  `sentence-transformers` for embeddings + cross-encoder reranking, `Chroma`
  for the vector store. All swappable via `config.py` once it exists.
- `mcp_server/` and `agents/reporting/` are kept as separate packages —
  the MCP **server** (Day 8) and the MCP **client** used inside the graph
  node (Day 9) are different processes and shouldn't be conflated.

## Open questions (resolved)

- LLM provider: config-driven via `LLM_PROVIDER=anthropic|openai` (`llm.get_chat_model()`);
  Anthropic remains the default. Both providers are exercised by the eval suite.
- Vector store: Chroma, persisted under `data/vectorstore/`. Postgres is used only as an
  optional checkpointer backend (`DATABASE_URL`), not for vectors.

## 2026-09-17 — Folder restructure

Aligned with a "production-ai-app" reference layout. Pure moves, no behaviour change:

- `agents/research/{retriever,reranker,query_rewriter}.py` → `retrieval/` (the research
  node now only orchestrates; retrieval components are reusable outside it).
- `validation.py` → `security/input_validation.py` (future prompt-injection defenses land here).
- `supervisor/` → `agents/supervisor/` (the router is an agent-layer concern).
- `ui/` → `frontend/`.
- `tests/` grouped by area: `agents/`, `retrieval/`, `ingestion/`, `evaluation/`, `graph/`,
  `checkpointing/`, `mcp/`, `security/`, `scripts/`.
- Added `AGENTS.md` (pointer to `CLAUDE.md`) and `.claude/` agents, skills, and hooks.

## 2026-09-18 — Current state (documentation pass)

Snapshot of what exists, for readers arriving after the sprint:

- **Graph** (`graph.py`): supervisor → {research, analytics, reporting} with conditional edges
  from `agents/supervisor/router.py`; every node wrapped by `with_error_boundary`; `run_graph()`
  bounds recursion and catches `GraphRecursionError`.
- **Human-in-the-loop**: the reporting node calls `interrupt()` before writing; the CLI and the
  browser UI both resume the graph with `{approved, feedback}`.
- **Retrieval** (`retrieval/`): query rewriting → Chroma retrieval → cross-encoder rerank, fed by
  hierarchical chunking in `ingestion/`.
- **MCP**: `mcp_server/` exposes filesystem writes over stdio; the reporting agent is the client.
  Unit tests use the SDK's in-process session; one pipeline test spawns the real server.
- **Checkpointing**: SQLite by default, Postgres when `DATABASE_URL` is set
  (see `postgres_checkpointer.md`).
- **Evaluation** (`evaluation/`): golden dataset + `scripts/run_evals.py`, optional LangSmith
  datasets/experiments with LLM judges.
- **Verification**: 157 pytest tests (1 skipped), ruff clean. GitLab CI is currently disabled
  (`.gitlab-ci.yml` is empty); see README "Known limitations".

## 2026-09-22 — Next.js frontend replaced with a Gradio app

- Removed `frontend/` (Next.js + CopilotKit) entirely — it drove the graph by calling a
  `langgraph dev` deployment over HTTP/streaming, in a separate Node.js process per LLM provider.
- Added `gradio_app/` (launched via `scripts/run_gradio.py`): a Gradio app that runs the graph
  **in-process**, calling `build_production_graph(get_checkpointer())` and `run_graph()` directly
  — the same pattern `scripts/run_graph_cli.py` uses for the terminal flow. There is no separate
  API server and no dual-port provider setup; the app talks to whichever provider `LLM_PROVIDER`
  selects, the same as every other flow in this repo.
- The Reporting Agent's human-approval `interrupt()` is unchanged. The Gradio app's Approve/Reject
  buttons resume the paused thread with `Command(resume={"approved": ..., "feedback": ...})`,
  mirroring the CLI's prompt-driven resume.
- `langgraph.json` and the `graph` export are untouched — `langgraph dev` / LangGraph Studio still
  work exactly as before for debugging and time-travel; they're just no longer what the UI talks
  to.
