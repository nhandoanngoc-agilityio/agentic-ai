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
- **Human-in-the-loop**: the `report_review` node calls `interrupt()` before writing; the CLI and the
  browser UI both resume the graph with `{approved, feedback}`.
- **Retrieval** (`retrieval/`): query rewriting → Chroma retrieval → cross-encoder rerank, fed by
  hierarchical chunking in `ingestion/`.
- **MCP**: `mcp_server/` exposes filesystem writes over stdio; the reporting agent is the client.
  Unit tests use the SDK's in-process session; one pipeline test spawns the real server.
- **Checkpointing**: SQLite by default, Postgres when `DATABASE_URL` is set
  (see `postgres_checkpointer.md`).
- **Evaluation** (`evaluation/`): golden dataset + `scripts/run_evals.py`, optional LangSmith
  datasets/experiments with LLM judges.
- **Verification**: 157 pytest tests (1 skipped), ruff clean. GitLab CI was disabled at the time
  (restored 2026-10-02, see below).

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

## 2026-09-29 — Agent versioning and trace stamping

`versioning.py` builds an agent manifest with five components — instructions
(system prompt hashes), model (provider, id, temperature, max_tokens), tools
(schema hash per tool, grouped by the agent allowed to call it, plus MCP limits),
knowledge (embedding/reranker models, chunking, retrieval k/top-n, index stamp)
and memory + safety (checkpointer, cache TTLs, loop/review limits, guardrail
pattern hashes). Each component is hashed; the hashes are hashed into a
12-char fingerprint. The version label is `<pyproject version>+<fingerprint>`.

Where it is stamped:

- Langfuse: trace `version` = label, flat metadata with every component hash
  and `git_sha`; `LANGFUSE_RELEASE` = git SHA (`graph.run_graph`, `observability.py`).
- Audit log: every line carries `agent_version` (`security/audit.py`).
- Evals: `data/eval_results/*.json` is now `{agent_version, git_sha, manifest,
  results}`; LangSmith experiments get the same metadata.
- Response cache: key includes the prompts + model hash, so a prompt edit
  invalidates it without bumping `CACHE_POLICY_VERSION`.

`scripts/show_agent_manifest.py --diff <eval file>` shows which components
changed since a given eval run. Git SHA is provenance only (from `GIT_SHA` /
`CI_COMMIT_SHA`, else `git rev-parse`); it is not in the fingerprint.

## 2026-09-29 — Eval release gate

`scripts/run_evals.py` now ends in a release verdict. Each provider's run
writes to its own `data/eval_results/<run id>/<provider>/` (audit log and
reports redirected there), so metrics cover only that run.
`evaluation/metrics.py` turns results plus that audit log into task success,
quality (judge mean), tool accuracy, safety, p95 latency and cost per graph
run. `evaluation/gate.py` checks them against `evals/gate.toml` thresholds and,
when present, the committed `evals/baseline.json` (tolerances and per-case
regressions). Full-pipeline and safety graph runs use an in-memory
checkpointer and answer the approval interrupt themselves
(`evaluation/graph_runs.py`). Safety cases live in `evaluation/safety_eval.py`.
Judge prompts are shared by the offline and LangSmith paths
(`evaluation/judges.py`), and both use the pinned judge model from
`gate.toml`. The baseline stores a judge id (quality source, judge model,
prompt hash) so a judge change is flagged. `--cache-regression` scopes the
cold and warm passes separately and gates only the warm one.

## 2026-09-30 — Production feedback loop

`feedback/` closes the loop from production into the release gate. `signals`
groups audit events per thread; `langfuse_source` adds thumbs-down
`user_feedback` scores (sent by Gradio) and ERROR observations; `triggers`
flags thumbs-down, reviewer rejections, run errors, blocked inputs, dropped
chunks, fallbacks and Langfuse errors. `harvest` writes scrubbed, deduplicated
candidates; a curator fills in declarative expectations and promotes them (CLI
or the Gradio "Regressions" tab) into `evals/regressions.jsonl`.
`evaluation/regression_eval.py` runs those as pass/fail `regression` cases
(`must_block` via the input guard only; the rest through the real graph).
`retention` prunes audit lines and checkpoint threads older than 90 days at app
start (daily), harvesting first.

## 2026-10-02 — Reporting split into draft and review nodes

LangGraph replays an interrupted node from the top on resume. With drafting and the approval
`interrupt()` in one node, every resume re-ran the LLM, so the file written after "Approve" was a
fresh draft nobody had reviewed, and round N cost N drafts. Now `reporting` drafts (plus output
guardrails) and stores `report_draft` in state; `report_review` holds only the interrupt and the
MCP write. Rejection routes `report_review -> reporting`; approve, discard, or the third rejection
return to the supervisor. The interrupt payload and the `{approved, feedback, discard}` resume
shape are unchanged, so the CLI, Gradio and eval harness needed no changes.

## 2026-10-02 — GitLab CI restored

`.gitlab-ci.yml` is back (it had been empty since 2026-09-14). One job on `python:3.11-slim`:
CPU-only torch, `pip install -e ".[dev]"`, then `ruff check`, `ruff format --check` and
`pytest -q` over `src tests scripts gradio_app`, with a JUnit report artifact. Only the
hermetic suite runs; the paid eval gate stays manual.

## 2026-10-02 — Targeted re-research

The supervisor could hand work back to Research, but Research only ever saw the objective: a
second pass retrieved roughly the same chunks and replaced the first pass's findings, so
hand-backs (three in the 2026-09-18 live run) cost LLM calls without adding anything.

- The supervisor now sees a short summary of each finding (`_FINDING_SNIPPET_CHARS`), and when
  it picks Research it returns `research_focus`, the gap to fill (`SupervisorRoute` from
  `decide_route`; `decide_next_step` still returns just the node).
- Research passes the gap to `rewrite_and_expand(..., focus=)` and reranks against objective +
  gap, then `merge_findings` unions the passes (dedupe by source + content, best score first,
  capped at `_MAX_FINDINGS` = 10).
- A hand-back that adds no new finding sets `research_exhausted`; `decide_route` then drops
  Research from the allowed options.
- An empty first pass (nothing above the rerank floor) sets `error` and ends the run. Before,
  `decide_next_step` sent a run with no findings back to Research every time, until the
  recursion limit.

Supervisor and query-rewriter prompts changed, plus two new limits in the manifest, so the
agent version moved: the OpenAI baseline needs a `--repeats 3 --update-baseline` re-run.

## 2026-10-05 — Checkpointer lifecycle and sync-safe MCP writes

- `get_checkpointer()` used to open a new connection on every call (the app and the startup
  retention prune each got one) and never close any; Postgres also called
  `from_conn_string(...).__enter__()` with no matching exit. It now returns one process-wide
  saver, rebuilt only if the configured target changes, and closes it at exit
  (`close_checkpointer`, registered with `atexit`). Postgres uses a 1–4 connection
  `psycopg_pool.ConnectionPool` with the same connection settings as `from_conn_string`;
  `psycopg-pool` is now listed in the `prod` extra.
- `report_review_node` wrote the report with `asyncio.run(...)`, which raises when an event loop
  is already running in the thread (an async caller, a notebook, `graph.ainvoke`). It now uses
  `async_utils.run_coroutine_sync`, which runs the coroutine on a worker thread's own loop in
  that case. The node stays synchronous on purpose: an `async def` node would break the
  `graph.invoke` calls the CLI, Gradio and evals make. The MCP safety eval uses the same helper.

## 2026-10-05 — Langfuse harvester on current read APIs

Langfuse returns 410 `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION` for `GET /api/public/v2/scores`
and the `traces` endpoints to organizations created on or after 2026-09-16, so the harvest
silently fell back to the audit log. `feedback/langfuse_source.py` now reads thumbs-down ratings
from `scores_v3.get_many_v3` (`NUMERIC`, `value_max=0`, `fields="subject"`; the subject says
whether a score sits on a session, trace or observation) and replaces `trace.get`/`trace.list`
with `observations.get_many(trace_id=... | session_id=..., fields="core,basic,metadata")`: trace
metadata such as `objective` is propagated onto every observation. Checked live against the
project (SDK 4.15.4). The first two regression cases were promoted the same day.

## 2026-10-05 — LangSmith judging honours `--repeats`

`run_evals.py --langsmith --repeats 3` repeated the deterministic cases three times but ran each
LangSmith experiment once, so judged quality rested on 7 rows (one each for reporting, analytics
and full pipeline). Two runs of the same agent scored 0.849 and 0.760; the whole gap was the
judge rating two near-identical reports 0.70 and 0.38. `run_langsmith_eval(repeats=)` now passes
`num_repetitions` to `evaluate`, so quality averages 21 rows. Baselines promoted before this are
single-sample. A checkpointer change (`sqlite` -> `postgres`) in "Changed since baseline" is
informational: evals always use an in-memory checkpointer.

## 2026-10-06 — Live progress in Gradio, and a Discard button

`run_graph(..., on_step=callback)` streams the graph (`stream_mode=["updates", "values"]`),
calls the callback after each node, and returns exactly what `invoke` returns, `__interrupt__`
included (tested against the real graph). Without a callback it still invokes, so the CLI, evals
and existing tests are unchanged; a failing callback is logged and never breaks a run. The
Gradio handlers are generators: `run_with_progress` runs the whole run on one worker thread
(per-thread context such as Langfuse tracing stays intact) and feeds step lines back through a
queue to a "Working…" chat turn. The approval row gained Discard (`{"discard": True}`), which the
graph already supported for the comparison UI.

