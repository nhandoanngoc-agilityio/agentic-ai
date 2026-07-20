# Market & Competitor Research Analyst Team

A multi-agent LangGraph demo system that orchestrates a supervisor and three
specialized sub-agents to research a market, analyze the findings, and write
a final report to disk via a custom local MCP server.

## Architecture

```
                     ┌───────────────────┐
          ┌─────────▶│  Supervisor Node  │◀─────────┐
          │          └─────────┬─────────┘          │
          │                    │ routes              │
          │        ┌───────────┼───────────┐         │
          ▼         ▼           ▼          ▼         │
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
   │  Research   │ │  Analytics  │ │  Reporting  │───┘
   │   Agent     │ │    Agent    │ │    Agent    │
   └─────────────┘ └─────────────┘ └──────┬──────┘
     RAG: rewrite    Python math/          │ MCP client
     → retrieve →    stat tools            ▼
     rerank                          ┌─────────────┐
                                     │ Local MCP    │
                                     │ FS Server    │
                                     └──────┬──────┘
                                            ▼
                                       reports/*.md
```

- **Supervisor Router Node** — orchestrates state flow between sub-agents via
  conditional edges.
- **Research Agent Node** — advanced RAG: query rewriting/expansion, vector
  retrieval, cross-encoder reranking.
- **Analytics Agent Node** — tool-calling agent using native Python
  math/stat functions to evaluate metrics.
- **Reporting Agent Node** — MCP client that calls a custom local MCP server
  to write markdown reports to the filesystem.

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — state machine / agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) + `langchain-anthropic` — LLM layer (Claude)
- `sentence-transformers` — local embeddings + cross-encoder reranking
- `langchain-chroma` / `chromadb` — local vector store
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) + `langchain-mcp-adapters` — filesystem-write tool server/client
- `langgraph-checkpoint-sqlite` (dev) / `langgraph-checkpoint-postgres` (prod) — durable checkpointing

See [`pyproject.toml`](pyproject.toml) for exact pinned versions.

## Project layout

```
src/market_research_team/
├── graph.py                # assembles + compiles the top-level StateGraph
├── state.py                # AgentState TypedDict schema
├── config.py                # pydantic-settings: model, paths, thresholds
├── supervisor/
│   └── router.py            # supervisor node + conditional-edge logic
├── agents/
│   ├── research/             # query rewriting, retrieval, reranking
│   ├── analytics/             # math/stat tool-calling agent
│   └── reporting/             # MCP client node
├── ingestion/                # loaders, hierarchical/recursive chunking, index build
├── mcp_server/                # local MCP server exposing filesystem write ops
└── checkpointing/             # memory / sqlite / postgres checkpointer factory

data/          # raw → processed documents, persisted vector store
reports/       # markdown reports written by the MCP server
scripts/       # one-shot CLI utilities (seed vectorstore, smoke test)
tests/         # pytest suite
docs/          # architecture notes
```

## 10-Day

### Week 1 — Graph Scaffolding & Advanced RAG Pipeline
| Day | Focus |
|---|---|
|   | Workspace setup, `pyproject.toml`, `langgraph.json` |
|   | Unified `AgentState` schema + mock graph routing nodes |
|   | Data ingestion pipeline: hierarchical/recursive chunking |
|   | Vector store retrieval loop + query rewriting/expansion |
|   | Cross-encoder reranking on Research Agent retrieval |

### Week 2 — Agent Tools, MCP Integration & Productionization
| Day | Focus |
|---|---|
|   | Analytics Agent tool-calling: native Python math/stat functions |
|   | Supervisor Node: conditional edges, Research ↔ Analytics handoff |
|   | Custom local MCP server: filesystem write operations |
|   | Reporting Agent ↔ MCP client wiring → markdown report output |
|   | Production guardrails: recursion limits, error boundaries, Postgres checkpointer, end-to-end test in LangGraph Studio |

## Getting started

```bash
# install (once pyproject.toml dependencies are finalized)
pip install -e ".[dev]"

# run the graph in LangGraph Studio
langgraph dev
```
## Status
