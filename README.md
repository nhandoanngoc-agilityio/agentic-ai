# Market & Competitor Research Analyst Team

A multi-agent LangGraph demo system that orchestrates a supervisor and three
specialized sub-agents to research a market, analyze the findings, and write
a final report to disk via a custom local MCP server.

## Architecture

```
               objective → input_guard → planner (2–4 sub-questions)
                                │
                     ┌──────────▼────────┐
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

- **Planner Node** — breaks the objective into 2–4 sub-questions (one company
  per question, none comparing companies) before any research. If planning fails, the plan
  is the objective itself, and the run proceeds as a single-question run.
- **Supervisor Router Node** — routes between sub-agents via conditional edges.
  On each LLM decision it first judges which plan items the findings answer
  (an "answered" claim must cite a source that is really in the findings), then
  hands back to Research for an open item, moves on to Analytics, or to
  Reporting. Each sub-question gets one targeted Research pass, and Research is
  offered only while an open item hasn't had it; Analytics is offered again only
  when the findings changed since it ran. A visit cap is the last safety net. Every route carries a rationale and says who decided it.
- **Research Agent Node** — advanced RAG: query rewriting/expansion, vector
  retrieval, cross-encoder reranking. On a hand-back it searches for the named
  plan item, skipping chunks it already holds and ranking against the gap, and
  merges the new findings in (deduplicated, capped at 10, new ones kept). A
  targeted pass that finds nothing new marks that item unanswerable; if
  nothing clears the rerank floor on the first pass, the run ends with an error
  instead of retrying.
- **Analytics Agent Node** — tool-calling agent using native Python
  math/stat functions to evaluate metrics; every reported number is backed
  by a real tool call, and each call's inputs are checked against the findings
  (an input the findings don't contain is flagged and grounds nothing). Each
  call carries a short label that names the metric in the report.
- **Reporting Agent** — two nodes. `reporting` drafts a markdown report and
  runs the output guardrails; `report_review` pauses on a human-approval
  interrupt, then calls a custom local MCP server (spawned over stdio) to write
  that exact draft to disk. A rejection loops back to `reporting` for a redraft.
  Plan items left unanswered are listed under "Open questions" in the draft. If the
  output check finds figures it can't trace to the evidence, the agent redrafts once
  before the reviewer sees the draft (a `self_check_redraft` event). Each run writes
  `<objective slug>-<run id>.md`, and the MCP server never replaces a different
  existing report, so re-running an objective keeps earlier approved reports.

Every node is wrapped with an error boundary (`guardrails.py`): an
unexpected failure is recorded into state, labelled with its exception type and
whether it was transient (timeout, rate limit, connection), and ends the run
cleanly instead of crashing. Model calls have a timeout and SDK-level retries;
the MCP write has a timeout. The safe entrypoint (`run_graph` in `graph.py`)
also bounds recursion and catches `GraphRecursionError`, including on a resume. A run that
failed on one step can be retried from its checkpoint (`run_graph_cli.py --retry <thread-id>`,
`graph.retry_failed_run`): only the failed step re-runs, and plan, findings, analytics and any
draft awaiting its write are kept.

### How the agent decides

| Decision | Who makes it |
|---|---|
| The sub-questions to answer | The model (planner), with a one-item fallback |
| Which sub-questions the findings answer | The model, checked by code against the findings' sources |
| Research again, analyze, or report | The model, among the steps code allows |
| Which sub-question to research next | The model (`focus_id`), defaulting to the first open item |
| A sub-question is unanswerable | Code: a targeted pass added nothing |
| End on error or discard, first research pass, visit cap, finish after the write | Code rules |
| Stop gathering when the run's token budget is spent, and report what it has | Code rule (`RunPolicy.max_run_tokens`) |
| Which math tool, with which inputs | The model (Analytics), inputs checked against the findings |
| Whether the report is written | A person (approval interrupt) |
| What earlier reviewers asked for, carried into new drafts | Long-term memory (`memory.py`, LangGraph store): reviewer feedback from past runs, sanitized when stored |

Each supervisor decision is logged as a `route_decision` audit event with
`decided_by` (`rule`, `llm` or `fallback`) and a rationale, and shown in the run's
progress messages.

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
├── observability.py          # optional Langfuse tracing + local token-usage tracking
├── versioning.py             # agent version = release + fingerprint of prompts/model/tools/limits
├── async_utils.py            # run an async call (MCP) from a sync node, loop or no loop
├── retrieval/                # query rewriting, vector retrieval, cross-encoder reranking
├── security/                 # layered guardrails: input validation, output filters, audit log
├── caching/                  # retrieval / rerank / opt-in response caches (SQLite)
├── agents/
│   ├── planner/              # objective -> 2-4 sub-questions (the plan)
│   ├── supervisor/           # plan coverage + routing, targeted hand-backs, rationale
│   ├── research/             # research node (uses retrieval/), merges passes
│   ├── analytics/            # native Python math/stat tool-calling agent
│   └── reporting/            # draft node + human-review node + MCP client wiring
├── ingestion/                # loaders, hierarchical/recursive chunking, index build
├── mcp_server/               # local MCP server exposing filesystem write ops
├── checkpointing/            # one shared SQLite / Postgres checkpointer per process
├── evaluation/               # golden dataset, eval harness, LLM judges, release gate
└── feedback/                 # production failures -> regression candidates, retention

gradio_app/    # Gradio UI (Research, Reports, Regressions tabs), runs the graph in-process
data/          # raw documents (sample corpus committed), vector store, checkpoints, eval results
evals/         # release gate config, approved baseline, curated regression cases (committed)
reports/       # markdown reports written by the MCP server
scripts/       # setup, seeding, CLI, Gradio, evals, manifest, harvest/promote/prune
tests/         # pytest suite, grouped by area (agents/, retrieval/, evaluation/, graph/, ...)
docs/          # architecture log, security, Postgres guide (design specs/plans stay local)
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
| Live end-to-end validation against real API credentials (Anthropic and OpenAI) |
| Config-driven LLM provider (`LLM_PROVIDER=anthropic\|openai`) instead of a hardcoded model |
| Prompt evaluation regression suite: golden dataset + real-LLM harness, distinct from the hermetic `pytest` suite |
| `scripts/setup_env.py`: one-shot install + fail-fast environment validation |

### Later — Operating it in production
| Focus |
|---|
| Gradio UI replacing the Next.js frontend: run, approve/reject, browse reports, rate runs |
| Layered guardrails (input, retrieval, tool, output, policy) with an audit log ([docs/security.md](docs/security.md)) |
| Retrieval, rerank and opt-in response caching with safe-caching rules |
| Langfuse tracing; agent versioning stamped on traces, audit lines and eval results |
| Release gate: thresholds, tolerances and an approved per-provider baseline (`evals/`) |
| Feedback loop: production failures harvested into regression candidates, promoted by a person |
| 90-day retention for the audit log and checkpoints, harvesting first |
| Reporting split into draft + review nodes, so the approved draft is exactly what gets written |
| Targeted re-research: the supervisor names the gap, Research merges passes, dead loops stop |
| One shared, pooled checkpointer per process, closed at exit |

## Results

Verified against this repo's current state, not aspirational:

- **Test suite (2026-10-06)**: 639 `pytest` tests: 636 pass, and the 3 live-Postgres tests skip
  unless `DATABASE_URL` points at a running server. Coverage 94.2% (floor 93%). `ruff check`,
  `ruff format --check` and `basedpyright` are clean over `src tests scripts gradio_app`.
  Hermetic: no API key or seeded vector store required. GitHub Actions (`.github/workflows/ci.yml`)
  runs all of these through
  `scripts/ci_checks.sh`; `scripts/ci_local.sh` runs the same job locally on a clean copy of the
  repo (Python 3.11, fresh `.venv`, no `.env`, minimal environment) before you push. Use
  `--worktree` to include uncommitted changes.
- **Release gate, approved baseline (2026-10-07, 10:05 UTC)**: `run_evals.py --provider openai
  --langsmith --repeats 3 --update-baseline` on `harness-polish` (agent `0.1.0+ac2f6794297e`:
  report self-check, no report overwrite, RunPolicy). Gate passed: task success 0.976, safety
  1.0, tool accuracy 1.0, judged quality 0.804. 11.4 model calls per graph run (supervisor
  3.6, query rewriter 3.2, analytics 2.3, report draft 1.3, planner 1.0), cost about $0.0027
  per run, median run 28.8 s, p95 58.5 s (cap 120 s). The self-check redraft fired in 3 of 9
  graph runs; a redraft roughly doubles the report step, which accounts for the two slowest
  runs (52 s, 58.5 s; the rest 24–34 s). The one failure is the known-unstable
  `supervisor_decision/plan_covered_analysis_done` (1 of 3 repeats).
- **Previous baseline (2026-10-07, 08:30 UTC)**: `run_evals.py --provider openai
  --langsmith --repeats 3 --update-baseline` after the targeted-research fix and the
  work-based gate, with model usage counted for every call. Gate passed: task success 1.0,
  safety 1.0, tool accuracy 1.0, judged quality 0.822. 11.7 model calls per graph run
  (supervisor 4.0, query rewriter 3.7, analytics 2.0, planner 1.0, report draft 1.0), cost
  about $0.0027 per run (the first figure that counts every call), median run 27.7 s, p95
  44.5 s (absolute cap 120 s). Recorded in `evals/baseline.json`.
- **Previous baseline (2026-10-07, 03:43 UTC)**: `run_evals.py --provider openai --langsmith
  --repeats 3 --update-baseline`, agent version `0.1.0+5b4a9c54be7d` (planner and
  coverage-driven supervisor; `gpt-4o-mini`, judge `gpt-5.4-mini`). Gate passed: task success
  0.952, safety 1.0, tool accuracy 1.0, judged quality 0.848 (analytics 1.00, supervisor 0.94,
  full pipeline 0.87, reporting 0.79, query rewrite 0.73), p95 latency 26.4 s. Its cost figure
  (about $0.00045 per run) counted only the report draft: until 2026-10-07 usage was recorded
  only for plain model calls, so the planner, supervisor, query rewriter and analytics calls
  were missing (see `docs/architecture.md`); the real cost per run is higher. Planning,
  trajectory and both regression cases pass in all 3 repeats. The one
  unstable case is `supervisor_decision/plan_covered_analysis_done` (1 of 3 here, 3 of 3 in the
  run before): gpt-4o-mini sometimes keeps a contract-value range open "to confirm" it.
  Getting there took two failed gate runs, each diagnosed from the run's audit log: a research
  loop, then an analytics loop (see `docs/architecture.md`, 2026-10-06). The LangSmith
  per-category `pass_rate` it prints counts a row as passed only if the judge scored exactly
  1.0, so judged categories can read 0.00; the gate uses the judge mean instead.
- **Postgres checkpointer (2026-10-05)**: with `DATABASE_URL` pointing at a local Postgres, all
  11 `tests/checkpointing` tests pass, including the 3 live ones: the real graph saves its
  checkpoints to Postgres, resumes through the approval interrupt, and keeps threads separate.
- **Previous baseline (2026-10-06, before the planner)**: `run_evals.py --provider openai --langsmith
  --repeats 3`, agent version `0.1.0+b6428b4bd1cc` (`gpt-4o-mini`, judge `gpt-5.4-mini`). Gate
  passed: task success 1.0, safety 1.0, tool accuracy 1.0, judged quality 0.892 averaged over
  21 judged rows (analytics 1.00, supervisor 0.96, full pipeline 0.90, reporting 0.86, query
  rewrite 0.79), p95 latency 27 s, about $0.0004 per run. Both regression cases pass in all
  3 repeats. Recorded in `evals/baseline.json`.
  Query rewriting rose from 0.48 (2026-10-05) after the rewriter was told to give each query
  its own facet and one company, instead of restating "X vs Y" comparisons that no document
  matches. Baselines before 2026-10-05 (0.83–0.85) came from a single judged sample per case,
  before `--repeats` reached the LangSmith judge, so they aren't comparable.

Earlier live runs, kept for the record (agent versions before versioning existed):

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

- **First prompt evaluation run** (`python scripts/run_evals.py`, real OpenAI credentials):
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
pytest -q                            # hermetic: no API key, no network
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

**Using [uv](https://docs.astral.sh/uv/) instead** (the repo commits `uv.lock`):

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # or ".[dev,prod]"
```

A venv created by `uv` has no `pip` of its own, so a plain `pip install` inside it silently
uses whichever `pip` is first on `PATH` (often the system one) and the packages never reach
`.venv`. In a `uv` venv, always install with `uv pip install`. A symptom of getting this wrong:
`No module named 'langgraph.checkpoint.postgres'` with `DATABASE_URL` set.

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
ruff check src tests scripts gradio_app
basedpyright        # type check against .basedpyright/baseline.json
pytest --cov        # coverage report (94% when adopted; fails under 93%)
```

`basedpyright` runs in basic mode from `[tool.pyright]` in `pyproject.toml`, which VS Code's
Pylance also reads. The ~230 errors that existed when it was adopted are recorded in
`.basedpyright/baseline.json` and fixed over time; the command fails only on new ones.

The test suite (513 tests) is hermetic — LLM calls and the vector store are
faked or run against real-but-local fixtures, so `pytest` doesn't require an
API key or the seeded vector store from step 3. The MCP server tests
(`tests/mcp/test_mcp_server.py`) use the SDK's in-process client session, so
no subprocess is spawned there; the reporting pipeline test
(`tests/agents/test_reporting_pipeline.py`) does spawn the real MCP server over
stdio and writes a real file to a temp dir, and `test_checkpointing.py` runs a
real SQLite round-trip — no network or API key needed for any of it.

```bash
ruff format --check src tests scripts gradio_app   # formatting
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
   limit. A hand-back to Research names the information still missing; an objective the
   documents can't answer ends after one Research pass with a "no relevant material" error.
3. Before the report is written, the Reporting Agent interrupts and prints the draft. Answer `y`
   to approve, or `n` plus optional feedback to request a redraft (up to the configured number of
   review rounds).
4. On approval, the report is written through the local MCP filesystem server to `reports/`
   (or `REPORTS_DIR`).

Each run is checkpointed under its thread id (SQLite by default, Postgres when
`DATABASE_URL` is set; see [docs/postgres_checkpointer.md](docs/postgres_checkpointer.md)), and
the approval prompt resumes it within the same CLI session. `--thread-id` must be new: the CLI
refuses an id that already has a run, since a fresh run on an old thread would inherit its
state. Every run spends real API money — the test suite does not.

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
python scripts/run_evals.py --repeats 3       # release check, ~3x the cost
```

Categories: query_rewrite, retrieval, supervisor_decision, analytics, tool_selection,
reporting, full_pipeline, safety, and regression (curated production failures, see
[Production feedback loop](#production-feedback-loop)). A timestamped JSON report goes to
`data/eval_results/<run id>/results.json`. The full-pipeline case needs the vector store seeded
(step 3 above). See `src/market_research_team/evaluation/golden_dataset.py` to add cases.

### Release gate

Every run ends with a gate verdict per provider: task success, judged quality, tool accuracy,
safety (must be 100%), model calls per run, p95 latency and cost per run, checked against
absolute floors and against the last approved baseline for that provider.

- Work, not wall-clock time, is compared with the baseline: model calls per graph run (+25%)
  and cost per run (+20%) move when the agent loops or takes longer routes, and stay put when
  the provider is slow. Run time is bounded by the absolute p95 cap; a median run time more
  than 25% above the baseline is reported as a warning, with the model-call change beside it.
  The gate output prints model calls per run by component for the candidate and the baseline.

- `evals/gate.toml` holds the thresholds, tolerances, the pinned judge model and per-model
  prices. A model without a price stops the run before it spends anything.
- `evals/baseline.json` holds the approved metrics per provider, stamped with the agent version.
  Only `run_evals.py --repeats 3 --update-baseline` writes it, and only if the gate passed;
  committing it is the promotion.
- Exit codes: `0` gate passed, `1` gate failed, `2` config or prerequisite error.
- Compare like with like: a baseline approved with `--langsmith` should be re-checked with
  `--langsmith`, or the quality numbers come from a different judge (the gate warns).
- `--repeats N` applies to the LangSmith judge as well (`num_repetitions`), so judged quality
  is an average of N samples per case. With one sample, a single judge call could move overall
  quality by about 0.05, the whole tolerance.
- `python scripts/show_agent_manifest.py` prints the current agent version and what it is made
  of, with no LLM calls.

`--langsmith` requires `LANGSMITH_API_KEY` (see [Configuration](#configuration)). It layers
LLM-judge scoring on top of — not instead of — the deterministic checks above: each of the 5
categories gets its cases mirrored into a LangSmith Dataset
(`market-research-team-<category>`) and scored in a LangSmith Experiment by both the existing
deterministic check and a category-tailored LLM judge (relevance / groundedness / appropriateness
depending on category). See `src/market_research_team/evaluation/langsmith_eval.py`.

## Gradio UI

A Gradio chat app (`gradio_app/`) that runs the graph in-process — no separate `langgraph dev`
deployment required. Gradio is a core dependency, so the step 1 install is enough:

```bash
python scripts/run_gradio.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`). It has three tabs:

- **Research** — submit an objective and watch the graph run against the configured
  `LLM_PROVIDER`: a "Working…" turn lists each step as it finishes (objective checked, research
  counts, analytics, routing), and stays in the chat as "Run steps" afterwards. The Reporting
  Agent's draft pauses for human approval exactly as it does for the CLI
  (`scripts/run_graph_cli.py`); Approve, Reject and Discard resume the same in-process run via
  `Command(resume=...)` on the paused thread. Reject takes optional feedback for a redraft;
  Discard ends the run without writing a report. Per-entity comparison charts are built from the run's analytics results once it
  finishes.
  After a run ends, a thumbs up/down rating is recorded for the feedback loop.
- **Reports** — browse markdown reports already written to `reports/` (or `REPORTS_DIR`).
- **Regressions** — harvest production failures into candidates, review them, and promote or
  reject each one (the same actions as `scripts/promote_case.py`).

Because the app calls `build_production_graph(get_checkpointer())` and `run_graph()` directly in
the same process (the same pattern as `run_graph_cli.py`), it needs only the one provider key that
pattern already requires — no dual-port setup, no `LLM_PROVIDER` juggling.

## Production feedback loop

Failures seen in real use become regression cases the release gate runs, with a person in the
middle:

```bash
python scripts/harvest_failures.py --since 7d     # thumbs-down, rejections, errors, blocked inputs -> candidates (free)
python scripts/promote_case.py list               # review candidates in data/regression_candidates/ (git-ignored)
python scripts/promote_case.py promote <id>       # -> evals/regressions.jsonl (committed), runs as `regression`
python scripts/prune_data.py                      # retention dry run; --apply deletes
```

Candidates are scrubbed of PII and secrets, and promotion refuses anything that still matches.
Nothing is promoted automatically. The audit log and checkpoints are kept for
`AUDIT_RETENTION_DAYS` (90) and pruned at most daily when the app or CLI starts, after failures
in them are harvested. Details: [docs/security.md](docs/security.md#retention).

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
| `AUDIT_RETENTION_DAYS` / `AUTO_PRUNE_ENABLED` | `90` / `true` | Retention window for the audit log and checkpoints; daily auto-prune at startup |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | unset | Pinned sampling parameters (provider default when unset); part of the agent version |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `60` / `2` | Per-call deadline and SDK retries (rate limits, 5xx, connection errors); part of the agent version |
| `MCP_WRITE_TIMEOUT_SECONDS` | `30` | Deadline for one MCP report write, including spawning the server |
| `RUN_POLICY__MAX_ROUTING_VISITS`, `__MAX_PLAN_ITEMS`, `__MAX_TOOL_ITERATIONS`, `__MAX_FINDINGS`, `__MAX_REVIEW_ROUNDS`, `__MAX_SELF_CHECK_REDRAFTS`, `__MAX_RUN_TOKENS`, `__MAX_REVIEWER_NOTES` | `8`, `4`, `4`, `10`, `3`, `1`, `50000`, `5` | Run limits, all in `RunPolicy` (`config.py`); part of the agent version |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | unset | Optional Langfuse tracing; a no-op unless both keys are set |

Other tunables (`anthropic_model` / `openai_model` names, defaults `claude-sonnet-5-5` /
`gpt-4o-mini`, embedding/reranker
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
| [docs/gradio_manual_test.md](docs/gradio_manual_test.md) | Manual test scenarios for the Gradio app, with inputs and expected results |
| `docs/superpowers/` (local only, git-ignored) | Working design specs and implementation plans; decisions that matter are summarized in `docs/architecture.md` |
| [gradio_app/](gradio_app/) | Gradio UI source — Research/Reports tabs, submit/approve/reject handlers |
| [CLAUDE.md](CLAUDE.md) | Conventions enforced in code, cost rules, agent/skill delegation for Claude Code |
| [.env.example](.env.example) | Every environment variable with a comment |

## Known limitations

Honest gaps, not hidden:

- **CI runs on GitHub Actions, not GitLab.** The company GitLab runner ran out of disk while
  installing dependencies (torch and friends), so `.gitlab-ci.yml` was removed on 2026-10-08 and
  the same job moved to `.github/workflows/ci.yml`. It runs only where the repository is pushed
  to GitHub. The real-LLM release gate stays a manual step: it costs money and needs API keys.
- **No Anthropic baseline.** `evals/baseline.json` has an approved baseline for OpenAI only, so
  an Anthropic eval run gets the absolute checks but no comparison against a baseline. Approving
  one needs a paid `run_evals.py --provider anthropic --repeats 3 --update-baseline` run.

- **Postgres is verified locally, not in production.** The checkpointer passed its live tests
  against a local Postgres on 2026-10-05 (see Results). It hasn't run against a managed
  Postgres or with several app instances sharing one database; see
  [docs/postgres_checkpointer.md](docs/postgres_checkpointer.md) Part 2 before doing that.
- **Merges aren't gated.** Work goes through feature branches, and `main` is protected against
  force-push, but nothing requires an approval or a passing CI run before a merge. On GitHub,
  a branch protection rule requiring the `CI / ruff, basedpyright, pytest` check would add that.
- **LangGraph Studio was validated via its API only.** `langgraph dev` was run and driven
  programmatically; the browser Studio UI itself (time-travel, manual state inspection) hasn't
  been clicked through interactively.
- **The Gradio UI has no automated browser test coverage.** `gradio_app/` is covered by pytest at
  the handler level (submit/approve/reject logic, chart building), but actually clicking through
  the running app is a manual check, the same way LangGraph Studio's browser UI is.

