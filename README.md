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
├── retrieval/                # query rewriting, vector retrieval, cross-encoder reranking
├── security/                 # input validation at agent boundaries
├── agents/
│   ├── supervisor/           # LLM-driven routing + Research/Analytics handoff
│   ├── research/             # research node (uses retrieval/)
│   ├── analytics/            # native Python math/stat tool-calling agent
│   └── reporting/            # drafts report + MCP client wiring
├── ingestion/                # loaders, hierarchical/recursive chunking, index build
├── mcp_server/               # local MCP server exposing filesystem write ops
├── checkpointing/            # SQLite / Postgres checkpointer factory
└── evaluation/               # golden dataset + offline/LangSmith prompt-eval harness

gradio_app/    # Gradio chat UI (Research + Reports tabs), runs the graph in-process
data/          # raw → processed documents, persisted vector store, checkpoint DB, eval results
reports/       # markdown reports written by the MCP server
scripts/       # setup_env.py, seed_vectorstore.py, run_graph_cli.py, run_evals.py, run_gradio.py
tests/         # pytest suite, grouped by area (agents/, retrieval/, evaluation/, graph/, ...)
docs/          # architecture notes, design specs and plans under docs/superpowers/
CLAUDE.md      # rules for Claude Code (AGENTS.md points other tools here); config in .claude/
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

### Post-sprint — Hardening & productionization
| Focus |
|---|
| Final documentation pass + step-by-step run instructions |
| GitLab CI: automated `ruff` + `pytest` on every push (currently disabled — see [Known limitations](#known-limitations)) |
| Live end-to-end validation against real API credentials (Anthropic and OpenAI) |
| Config-driven LLM provider (`LLM_PROVIDER=anthropic\|openai`) instead of a hardcoded model |
| Prompt evaluation regression suite: golden dataset + real-LLM harness, distinct from the hermetic `pytest` suite |
| `scripts/setup_env.py`: one-shot install + fail-fast environment validation |

## Results

Verified against this repo's current state, not aspirational:

- **Test suite**: 157 `pytest` tests passing (1 skipped), `ruff check src tests scripts` clean — hermetic,
  no API key or seeded vector store required.
- **Live end-to-end run** (real OpenAI credentials, `gpt-5-mini`, objective *"Assess Acme vs Globex
  pricing strategy and recommend a competitive positioning"*): completed in one pass via
  `scripts/run_graph_cli.py` — 5 research findings gathered across 3 Research Agent visits, 7
  analytics tool calls, one report written to disk through the real MCP filesystem server. The
  supervisor routed Research → Analytics → Research → Research → Reporting → FINISH, staying well
  under the recursion/visit cap. The generated report correctly derived numbers straight from the
  source documents with no hallucination, e.g.:

  > Globex deals are negotiated and bundled with mandatory implementation services. Industry
  > estimates place typical annual contract value (ACV) between $150K and $400K.
  > Mean ACV ≈ $275,000. ACV range = $250,000 (150K → 400K), a 166.7% increase from the low to
  > high end.

- **Live end-to-end run, re-validated 2026-09-18** (real OpenAI credentials, `gpt-5-mini`,
  objective *"Compare Acme and Globex go-to-market strategy and recommend where Acme should invest
  next"*, `--thread-id live-openai-20260918`): Research → Analytics → Reporting → FINISH in one
  pass. 4 rewritten queries, 12 candidates retrieved, 5 kept after reranking; 16 analytics tool
  calls; the human-approval interrupt fired before the write; the report was written through the
  real MCP server to `reports/compare-acme-and-globex-go-to-market-strategy-and-recommend-.md`.
  Every figure in the report was checked against `data/raw/`: 1,200 vs 400 customers, 340 vs 900
  employees, $150K–$400K ACV, 8–16 week implementations, founding years and HQ cities all match
  the source text.

- **Anthropic path, same date**: could not be validated live because no Anthropic key was
  configured in this environment. The run is still useful as a guardrail check: the Analytics
  node's authentication error was caught by the error boundary, recorded in `state.error`, and the
  graph ended cleanly with `report_path: None` instead of crashing. Re-run
  `LLM_PROVIDER=anthropic python scripts/run_graph_cli.py "..."` once `ANTHROPIC_API_KEY` is set.

- **Prompt evaluation regression suite** (`python scripts/run_evals.py`, real OpenAI credentials):
  7/7 cases passed. Notably `analytics/globex_acv_range` — the model called 7 real statistics
  tools, and every reported figure (min $150K, max $400K, mean $275K, range $250K) traced back to
  the source text rather than being invented, which is exactly the class of regression this suite
  exists to catch.

## Getting started

### Quick start (copy-paste)

The full path from a fresh clone to a written report, in order. Each step is explained in the
numbered sections that follow.

```bash
git clone <this-repo> && cd agentic-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                 # then set ANTHROPIC_API_KEY (or LLM_PROVIDER=openai + OPENAI_API_KEY)
python scripts/setup_env.py --skip-install   # validates packages + the key your provider needs
python scripts/seed_vectorstore.py   # builds data/vectorstore/ from data/raw/
pytest -q                            # 157 tests, no API key needed
python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy"   # real LLM calls
```

The CLI pauses before writing to disk and asks `Approve this write to disk? [y/N]`. Answer `y`
and the report lands in `reports/<slugified-objective>.md`.

### Prerequisites

- Python 3.11+ (`requires-python = ">=3.11"` in `pyproject.toml`; `langgraph.json` pins 3.11 for the dev server)
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

Once `.env` is set, `python scripts/setup_env.py` re-runs the install from
step 1 and then validates the result: it imports every critical package
(LangGraph, the configured LLM provider, Chroma, MCP, etc.) and confirms
`.env` has the API key `LLM_PROVIDER` actually needs, failing fast with an
itemized list instead of a cryptic error later. Use `--skip-install` to
validate an existing environment without reinstalling, or `--prod` to also
cover the Postgres checkpointer extra.

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

The test suite (157 tests) is hermetic — LLM calls and the vector store are
faked or run against real-but-local fixtures, so `pytest` doesn't require an
API key or the seeded vector store from step 3. The MCP server tests
(`tests/mcp/test_mcp_server.py`) use the SDK's in-process client session, so
no subprocess is spawned there; the reporting pipeline test
(`tests/agents/test_reporting_pipeline.py`) does spawn the real MCP server over
stdio and writes a real file to a temp dir, and `test_checkpointing.py` runs a
real SQLite round-trip — no network or API key needed for any of it.

```bash
ruff format --check src tests scripts   # formatting, same check CI would run
```

### 5. Run the graph

**Option A — CLI:**

```bash
python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy"

# with durable checkpointing (persists to data/checkpoints.sqlite by default):
python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy" --thread-id demo-1

# raise the step cap for a long objective (default comes from config.py):
python scripts/run_graph_cli.py "..." --recursion-limit 60
```

What happens during a run:

1. The objective is validated (`security/input_validation.py`) — empty or oversized input is
   rejected before any LLM call.
2. The supervisor routes between Research (RAG over the seeded index), Analytics (Python stat
   tools), and Reporting until it decides to FINISH, bounded by a visit cap and the recursion
   limit.
3. Before the report is written, the Reporting Agent interrupts and prints the draft. Answer `y`
   to approve, or `n` plus optional feedback to request a redraft (up to the configured number of
   review rounds).
4. On approval, the report is written through the local MCP filesystem server to `reports/`
   (or `REPORTS_DIR`).

Re-running with the same `--thread-id` resumes from the SQLite checkpoint instead of starting
over. Every run spends real API money — the test suite does not.

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

## Guardrails

One deterministic guardrail per layer, placed where it is cheapest. None of them call a
model, so they add no tokens and no measurable latency. Full table and rationale in
[docs/security.md](docs/security.md).

| Layer | What it does | Where |
|---|---|---|
| Input | Length bounds, control-char stripping, prompt-injection and exfiltration denylist, run as the first graph node so CLI, Studio and the Gradio UI all get it | `security/input_guard.py` |
| Retrieval | Cross-encoder score floor drops irrelevant chunks; injection scan drops poisoned ones | `retrieval/reranker.py`, `agents/research/node.py` |
| Tool | Fixed math functions only; MCP writes `.md` inside `reports/` with size caps | `mcp_server/fs_server.py` |
| Output | PII redaction, credential scrub, and `[unverified]` marks on figures that don't trace to evidence, shown as warnings at the approval prompt | `security/output_filters.py` |
| Policy | Append-only audit log per run (`data/audit.jsonl`), error boundaries, recursion and visit caps | `security/audit.py`, `guardrails.py` |

Every trigger is recorded as a `guardrail_events` entry in state; the CLI prints them at
the end of a run.

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
python scripts/run_evals.py --langsmith       # also run LangSmith dataset sync + LLM-judge experiments
```

Exits non-zero if any case fails, and writes a timestamped JSON report to
`data/eval_results/`. The full-pipeline case needs the vector store seeded
(step 3 above). See `src/market_research_team/evaluation/golden_dataset.py`
to add cases.

`--langsmith` requires `LANGSMITH_API_KEY` (see [Configuration](#configuration)). It layers
LLM-judge scoring on top of — not instead of — the deterministic checks above: each of the 5
categories gets its cases mirrored into a LangSmith Dataset
(`market-research-team-<category>`) and scored in a LangSmith Experiment by both the existing
deterministic check and a category-tailored LLM judge (relevance / groundedness / appropriateness
depending on category). See `src/market_research_team/evaluation/langsmith_eval.py`.

## Gradio UI

A Gradio chat app (`gradio_app/`) that runs the graph in-process — no separate `langgraph dev`
deployment required. Install the `ui` extra and launch it:

```bash
pip install -e ".[dev,ui]"
python scripts/run_gradio.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`). It has two tabs:

- **Research** — submit an objective and watch the graph run against the configured
  `LLM_PROVIDER`. The Reporting Agent's draft pauses for human approval exactly as it does for
  the CLI (`scripts/run_graph_cli.py`); Approve/Reject buttons resume the same in-process run via
  `Command(resume=...)` on the paused thread, with an optional feedback field driving a redraft on
  reject. Per-entity comparison charts are built from the run's analytics results once it
  finishes.
- **Reports** — browse markdown reports already written to `reports/` (or `REPORTS_DIR`).

Because the app calls `build_production_graph(get_checkpointer())` and `run_graph()` directly in
the same process (the same pattern as `run_graph_cli.py`), it needs only the one provider key that
pattern already requires — no dual-port setup, no `LLM_PROVIDER` juggling.

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
| `LANGSMITH_API_KEY` | unset | Required for `scripts/run_evals.py --langsmith` (dataset sync + LLM-judge experiments) |
| `VECTORSTORE_DIR` | `./data/vectorstore` | Chroma persistence directory |
| `REPORTS_DIR` | `./reports` | Where the MCP server writes markdown reports |
| `DATABASE_URL` | unset | If set, `get_checkpointer()` uses Postgres instead of the local SQLite file |
| `RERANK_SCORE_FLOOR` | `-8.0` | Cross-encoder logit below which retrieved chunks are dropped |
| `AUDIT_LOG_PATH` | `./data/audit.jsonl` | Append-only JSONL audit log of finished runs and human decisions |

Other tunables (`anthropic_model` / `openai_model` names, embedding/reranker
model names, chunk sizes, recursion limit, checkpoint DB path) have
sensible defaults in `config.py` and are generally not something you need
to touch to run the demo. Every node builds its LLM through
`llm.get_chat_model()` rather than importing a provider class directly, so
`LLM_PROVIDER` is the only thing that needs to change to switch providers.

## Documentation map

| Document | What it covers |
|---|---|
| [README.md](README.md) (this file) | Overview, setup, run instructions, configuration |
| [docs/architecture.md](docs/architecture.md) | Design log: layout decisions and the 2026-09 restructure |
| [docs/security.md](docs/security.md) | Guardrails per layer, what is deliberately absent, data handling |
| [docs/postgres_checkpointer.md](docs/postgres_checkpointer.md) | Standing up Postgres locally and what changes for production |
| [docs/superpowers/specs/](docs/superpowers/specs/), [docs/superpowers/plans/](docs/superpowers/plans/) | Design specs and implementation plans per feature |
| [gradio_app/](gradio_app/) | Gradio UI source — Research/Reports tabs, submit/approve/reject handlers |
| [CLAUDE.md](CLAUDE.md) | Conventions enforced in code, cost rules, agent/skill delegation for Claude Code |
| [.env.example](.env.example) | Every environment variable with a comment |

## Known limitations

Honest gaps, not hidden:

- **GitLab CI is currently switched off.** `.gitlab-ci.yml` was emptied on 2026-09-14 (commit
  `3fcc1fa`) while eval work was in progress and has not been restored. The previous pipeline ran
  `ruff check` and `pytest -q` on `python:3.11-slim`; `git show 3fcc1fa^:.gitlab-ci.yml` recovers
  it. Until then, run `pytest` and `ruff` locally before pushing.

- **The Postgres checkpointer is unverified against a real database.** `get_checkpointer()`
  supports `DATABASE_URL`, but it's only been exercised via code review, not a live Postgres
  instance.
- **No PR/branch review workflow.** All work has been committed directly to `main`. `main` is
  branch-protected against force-push, but nothing currently requires review before a merge.
- **LangGraph Studio was validated via its API only.** `langgraph dev` was run and driven
  programmatically; the browser Studio UI itself (time-travel, manual state inspection) hasn't
  been clicked through interactively.
- **The Gradio UI has no automated browser test coverage.** `gradio_app/` is covered by pytest at
  the handler level (submit/approve/reject logic, chart building), but actually clicking through
  the running app is a manual check, the same way LangGraph Studio's browser UI is.

