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

- **Supervisor Router Node** — LLM-driven routing between sub-agents via
  conditional edges; can hand work back to Research or Analytics, not just
  march forward, bounded by a visit cap.
- **Research Agent Node** — advanced RAG: query rewriting/expansion, vector
  retrieval, cross-encoder reranking.
- **Analytics Agent Node** — tool-calling agent using native Python
  math/stat functions to evaluate metrics; every reported number is backed
  by a real tool call.
- **Reporting Agent Node** — drafts a markdown report, then calls a custom
  local MCP server (spawned over stdio) to write it to disk.

Every node is wrapped with an error boundary (`guardrails.py`): an
unexpected failure is recorded into state and ends the run cleanly instead
of crashing. The safe entrypoint (`run_graph` in `graph.py`) also bounds
recursion and catches `GraphRecursionError`.

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — state machine / agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) + `langchain-anthropic` / `langchain-openai` — LLM layer (Claude by default, OpenAI via `LLM_PROVIDER=openai`)
- `sentence-transformers` / `langchain-huggingface` — local embeddings + cross-encoder reranking
- `langchain-chroma` / `chromadb` — local vector store
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) + `langchain-mcp-adapters` — filesystem-write tool server/client
- `langgraph-checkpoint-sqlite` (default) / `langgraph-checkpoint-postgres` (optional, via `DATABASE_URL`) — durable checkpointing

See [`pyproject.toml`](pyproject.toml) for exact pinned versions.

## Project layout

```
src/market_research_team/
├── graph.py                  # graph assembly, error-boundary wiring, run_graph() safe entrypoint
├── state.py                  # AgentState TypedDict schema
├── config.py                 # pydantic-settings: model, paths, thresholds
├── llm.py                    # chat model factory (LLM_PROVIDER: anthropic | openai)
├── guardrails.py             # per-node error-boundary wrapper
├── supervisor/
│   └── router.py             # LLM-driven routing + Research/Analytics handoff
├── agents/
│   ├── research/              # query rewriting, retrieval, cross-encoder reranking
│   ├── analytics/              # native Python math/stat tool-calling agent
│   └── reporting/              # drafts report + MCP client wiring
├── ingestion/                 # loaders, hierarchical/recursive chunking, index build
├── mcp_server/                 # local MCP server exposing filesystem write ops
├── checkpointing/               # SQLite / Postgres checkpointer factory
└── evaluation/                   # golden dataset + offline prompt-eval harness

data/          # raw → processed documents, persisted vector store, checkpoint DB, eval results
reports/       # markdown reports written by the MCP server
scripts/       # seed_vectorstore.py, run_graph_cli.py, run_evals.py
tests/         # pytest suite
docs/          # architecture notes
```

## What was built

### Week 1 — Graph Scaffolding & Advanced RAG Pipeline
| Focus |
|---|
| Workspace setup, `pyproject.toml`, `langgraph.json` |
| Unified `AgentState` schema + mock graph routing nodes |
| Data ingestion pipeline: hierarchical/recursive chunking |
| Vector store retrieval loop + query rewriting/expansion |
| Cross-encoder reranking on Research Agent retrieval |

### Week 2 — Agent Tools, MCP Integration & Productionization
| Focus |
|---|
| Analytics Agent tool-calling: native Python math/stat functions |
| Supervisor Node: conditional edges, Research ↔ Analytics handoff |
| Custom local MCP server: filesystem write operations |
| Reporting Agent ↔ MCP client wiring → markdown report output |
| Production guardrails: recursion limits, error boundaries, Postgres checkpointer, end-to-end test in LangGraph Studio |

## Getting started

### Prerequisites

- Python 3.11+
- An API key for one LLM provider — [Anthropic](https://console.anthropic.com/) (default) or [OpenAI](https://platform.openai.com/api-keys) — for the LLM calls made by query rewriting, supervisor routing, analytics, and report drafting

### 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

This installs the runtime dependencies plus `pytest`, `ruff`, and
`langgraph-cli` (for `langgraph dev` / Studio). To also exercise the
Postgres checkpointer, add the `prod` extra: `pip install -e ".[dev,prod]"`.

### 2. Configure environment

```bash
cp .env.example .env
```

Then edit `.env` and set `ANTHROPIC_API_KEY` (default provider) — or set
`LLM_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI instead. Everything
else has a working default — see [Configuration](#configuration) below.

### 3. Seed the vector store

The Research Agent retrieves against a local Chroma index built from the
sample competitor/market documents in `data/raw/`:

```bash
python scripts/seed_vectorstore.py
```

This chunks the documents hierarchically, embeds the leaf chunks, and
persists them to `data/vectorstore/` (plus a parent-section JSON docstore
under `data/processed/`). Re-run it any time you add or change files in
`data/raw/`.

### 4. Run the test suite

```bash
pytest
ruff check src tests scripts
```

The test suite (95 tests) is hermetic — LLM calls, the vector store, and
the MCP subprocess are all faked or run against real-but-local fixtures,
so `pytest` doesn't require `ANTHROPIC_API_KEY` or the seeded vector store
from step 3. A handful of tests do spawn the real local MCP server
subprocess (`test_mcp_server.py`, `test_reporting_pipeline.py`) and run a
real SQLite checkpointer round-trip (`test_checkpointing.py`) — no network
or API key needed for any of it.

### 5. Run the graph

**Option A — CLI:**

```bash
python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy"

# with durable checkpointing (persists to data/checkpoints.sqlite by default):
python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy" --thread-id demo-1
```

**Option B — LangGraph Studio:**

```bash
langgraph dev --no-browser   # drop --no-browser to auto-open Studio
```

This starts a local API server (default `http://127.0.0.1:2024`) and
registers the graph as the `market_research_team` assistant. Open the
printed Studio URL to run it interactively, inspect state at each step,
and time-travel through checkpoints. `langgraph dev` manages its own
persistence — it doesn't use `checkpointing/store.py`, which is for
standalone use outside the dev server (see `run_graph()` in `graph.py`).

## Prompt evaluation regression suite

`pytest` proves the *code* is correct (routing logic, retrieval merging,
tool execution, error boundaries) using fake LLMs — it never calls a real
model. `scripts/run_evals.py` proves the *prompts* are still behaving,
using real LLM calls against a small golden dataset grounded in the
sample documents (e.g. Acme's real $49/seat price, Globex's real
$150K–$400K ACV range) instead of synthetic expectations. It checks
structural/grounded properties (query count, keyword coverage, whether a
computed number actually derives from the source data) rather than exact
text, since LLM output isn't deterministic.

Run it after changing a system prompt, switching models, or before a
release — not on every commit, since it costs real API calls:

```bash
python scripts/run_evals.py                  # current LLM_PROVIDER
python scripts/run_evals.py --provider openai
python scripts/run_evals.py --compare         # anthropic AND openai, side by side
```

Exits non-zero if any case fails, and writes a timestamped JSON report to
`data/eval_results/`. The full-pipeline case needs the vector store seeded
(step 3 above). See `src/market_research_team/evaluation/golden_dataset.py`
to add cases.

## Configuration

All settings live in `src/market_research_team/config.py` (pydantic-settings)
and can be overridden via `.env` or real environment variables. From
[`.env.example`](.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` — selects which chat model `llm.get_chat_model()` builds |
| `ANTHROPIC_API_KEY` | *(required if `LLM_PROVIDER=anthropic`)* | Claude access for every LLM-driven node |
| `OPENAI_API_KEY` | *(required if `LLM_PROVIDER=openai`)* | OpenAI access for every LLM-driven node |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | off | Optional LangSmith tracing |
| `VECTORSTORE_DIR` | `./data/vectorstore` | Chroma persistence directory |
| `REPORTS_DIR` | `./reports` | Where the MCP server writes markdown reports |
| `DATABASE_URL` | unset | If set, `get_checkpointer()` uses Postgres instead of the local SQLite file |

Other tunables (`anthropic_model` / `openai_model` names, embedding/reranker
model names, chunk sizes, recursion limit, checkpoint DB path) have
sensible defaults in `config.py` and are generally not something you need
to touch to run the demo. Every node builds its LLM through
`llm.get_chat_model()` rather than importing a provider class directly, so
`LLM_PROVIDER` is the only thing that needs to change to switch providers.

