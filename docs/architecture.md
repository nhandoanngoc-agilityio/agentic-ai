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

## Open questions

- Confirm target LLM/provider if `langchain-anthropic` default is wrong.
- Confirm vector store choice (Chroma vs. Qdrant/pgvector) before Day 3-4 ingestion work.

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
