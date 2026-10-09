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

## 2026-10-06 — Planning, coverage-driven routing, and harness fixes

Follows an audit of the project as an agentic system: the harness was strong, but the
supervisor mostly picked among 2–3 options that code had already narrowed, and recorded no
reason. Changes, in the order they landed:

- **CI**: `.gitlab-ci.yml` had been emptied again (the 2026-10-02 entry above no longer held).
  Restored with `ruff check`, `ruff format --check`, `basedpyright` and `pytest --cov`.
- **Harness**: `run_graph` no longer crashes when a resume hits the recursion limit (it spread a
  `Command` as if it were state; basedpyright had flagged this and it sat in the baseline).
  One `new_run_state()` replaces three diverging builders. The visit cap reads a
  `supervisor_visits` counter instead of counting messages, and the CLI refuses a
  `--thread-id` that already has a run. Model calls get an explicit timeout and SDK retries,
  and the MCP write a timeout. The error boundary labels each failure with its exception type
  and whether it was transient, and the harvester tags transient ones `transient_error`.
  Retries stay at the call level on purpose: retrying a node would repeat tool calls already
  made, or the approval interrupt.
- **Grounding**: analytics tool arguments no longer count as evidence for the report (an
  invented input used to ground itself and its result). Inputs not found in the findings
  raise `ungrounded_tool_input`. Analytics now fences findings like Reporting
  (`security/fencing.py`). Tools take a `label`, so the report names metrics.
- **Planner + coverage-driven supervisor**: a `planner` node (`input_guard → planner →
  supervisor`) writes 2–4 sub-questions. Each LLM supervisor decision returns a rationale,
  coverage per item (validated against the findings' sources) and the item to research next.
  Research is offered only while an item is open. A targeted pass that adds nothing marks only
  that item unanswerable. Reporting lists unanswered items under "Open questions". Every route
  is a `route_decision` audit event with `decided_by` (`rule`, `llm`, `fallback`).
- **Evals**: the supervisor cases now each have one right answer (the old ones passed for any
  allowed choice). New `planning` and `trajectory` categories (the trajectory reuses the
  full-pipeline run and its `route_decision` events). The analytics case checks inputs are
  grounded, a reporting case checks reviewer feedback is acted on, and a safety case checks that
  invented tool inputs are flagged. Hermetic graph tests cover a transient failure mid-run and
  an MCP write timeout after approval.

The baseline in `evals/baseline.json` predates all of this; it needs a
`run_evals.py --repeats 3 --update-baseline` run (paid).

### Same day — first gate run of the planner: a research loop

The first `run_evals.py --repeats 3` after the planner failed the gate on baseline
`task_success` (0.905 vs 1.0) and `latency_p95_ms` (41.8 s vs 27.3 s, limit +25%). Every one
of the 9 graph runs went `research ×6` until the visit cap forced Analytics: no plan item was
ever judged answered, and the same item was often retargeted. Causes and fixes:

- The supervisor anchored on items shown as `[open]` and could skip `coverage` (it had a
  default). Open items are now shown as "to judge", `coverage` is required, and the prompt says
  an item is answered when the findings state its facts, estimates and ranges included. This
  was also the failing `plan_covered_analysis_done` case (2 of 3 repeats chose research).
- The supervisor saw 160 characters per finding; real chunks run to 800, so the answering
  figure was often cut off. Now 500.
- A sub-question only became unanswerable when a targeted pass added nothing, which a small
  corpus rarely produces. Each item now gets one targeted search (`attempted`); Research is
  offered only while an open item hasn't had it. This bounds research passes at plan items + 1
  regardless of the model's judgment.
- The planner wrote comparison and benchmark questions the corpus can't answer (the failing
  `planning` case). It now writes 2–4 single-document questions and never phrases comparisons.
- The trajectory check passed all of this. It now also fails when the visit cap fires or a
  sub-question is targeted twice.

### Same day — second gate run: an analytics loop

With research bounded, the next run (`task_success` 0.833, p95 42.3 s) showed every graph run
going `analytics(rule) > analytics(llm|fallback) [> analytics]` until the visit cap forced
Reporting. Analytics is the slowest node (5–9 s), so the repeats were most of the regression.

- Analytics is offered again only when the findings changed since it last ran (Research bumps
  `findings_version`; Analytics records `analyzed_findings_version`).
- The decision schema is built per call with `next` limited to the allowed steps, and an
  invalid choice now falls forward (`allowed[-1]`, as the exception path already did) rather
  than back to `allowed[0]`, which had re-run Analytics.
- With one step left, code still decides by rule, unless a plan item is open: then the model
  is asked (its choice limited to that step) so the last targeted search gets judged and a
  found answer isn't reported as an open question.
- The visit cap moved from 6 to 8: with at most 4 targeted searches and Analytics only on new
  findings, a run legitimately needs up to 7 decisions, so the cap is a safety net again.
- The supervisor prompt says a range or estimate is an answer (the model kept Globex's
  "$150K–$400K" open "for precision"). The supervisor eval now fails a route reached by
  fallback, and the planning check flags only questions phrased as comparisons, not a
  market question that names both companies as context.

## 2026-10-07 — Targeted searches return new chunks; CI reproducible locally

A live CLI run ("Compare Acme and Globex pricing and recommend a competitive positioning")
reported market size as unanswerable although `market_overview.md` covers it. The targeted
search retrieved 12 candidates, but ranked them against objective + gap: the pricing objective
put the already-held pricing chunks in all 5 top slots, the pass counted "0 new", and the item
was marked unanswerable.

- A targeted search now drops chunks already held before reranking (`exclude`), and ranks
  against the gap alone.
- "Unanswerable" means the targeted search found no relevant new chunk above the rerank floor,
  not that its chunks failed to survive the 10-finding cap.
- A targeted pass's findings are kept when merging (`keep_new`); the lowest-scoring held ones
  make room. Scores from different passes are ranked against different queries, so comparing
  them could drop exactly the chunk the pass was sent for.
- When code redirects a hand-back away from an item the model chose (already searched), the
  rationale says so instead of carrying the model's reasoning about the other item.

CI: the first push failed because `pyproject.toml` points basedpyright at `./.venv` (absent in
the job) and the job installed only `.[dev]`, so the Postgres imports were unresolved. CI now
creates `.venv`, installs `.[dev,prod]` and runs `scripts/ci_checks.sh`;
`scripts/ci_local.sh` runs the same script on a clean copy with an empty environment. That run
also exposed two tests passing only because the developer's shell exported `OPENAI_API_KEY`;
`tests/conftest.py` now gives every test dummy keys.

### Same day — release gate: latency compared by the median run

A gate run after the research fix failed only `latency_p95_ms` vs baseline (38.5 s vs 26.4 s,
+46%). Routes were shorter than in the baseline run, but every model call was slower, the
planner included (+36%, and the fix cannot touch it): provider response time. With 9 graph
runs, "p95" is the slowest run, and the baseline was itself one fast sample.

The baseline comparison now uses `latency_median_ms` (same tolerance, +25%); p95 stays as the
absolute cap (`latency_p95_ms_max`). A baseline recorded before the median existed is compared
by its p95 (looser) with a warning, until the next `--update-baseline` records a median.

Known limit: the median absorbs one slow run, not a slow hour. In the failing run the median
also moved (22.4 s to 31.7 s), so against a median baseline it would still have failed.


### Same day — the gate compares agent work, and model usage is counted for every call

The median comparison above still failed the slow-provider run against a median baseline, so
the gate now compares **work** with the baseline and treats time as a cap:

- New metrics: `model_calls_per_run` (mean model calls per graph run) and
  `model_calls_by_component`. Gated against the baseline at +25%
  (`model_calls_max_increase` in `gate.toml`), next to cost per run (+20%). Loops and longer
  routes move both; a slow provider hour moves neither.
- Wall-clock: p95 stays the absolute cap (120 s). The median run time vs the baseline became a
  warning that shows the model-call change beside it.

Building that exposed two older bugs in usage recording:

- `get_chat_model()` attached the usage callback with `.with_config(...)`. `bind_tools` and
  `with_structured_output` build new runnables from the model and drop config callbacks, so
  only plain `.invoke` calls were recorded: inside a graph run, the report draft alone. Every
  `cost_per_run_usd` and token summary before this counted only that call. The callback now
  sits on the model object (`callbacks=` in the constructor), which every wrapper reuses.
- The callback took `tags[0]` as the component, and inside a graph run LangGraph's own tag
  (`seq:step:1`) comes first. It now takes the first tag without a colon.

Because cost and call counts recorded before and after are not comparable, metrics carry
`usage_accounting` (1 before, 2 after). The gate compares cost and model calls with a baseline
only under the same accounting, and warns otherwise; the absolute cost cap still applies. The
next `--update-baseline` records version 2.

## 2026-10-07 — Harness polish: reports kept, self-check, run usage, RunPolicy

- **No silent overwrite.** Each run writes `<objective slug>-<run id>.md`, the run id a short
  hash of the thread id (stable across a resumed review, distinct between runs). The MCP
  `write_report` refuses to replace an existing report with different content unless called
  with `overwrite=true`; an identical rewrite (a review replayed after a crash) succeeds.
- **Self-check redraft.** When the output check marks figures it can't trace to the evidence,
  `reporting_node` redrafts once with a note naming them, before the reviewer sees the draft;
  what remains travels as warnings. Recorded as a `policy / self_check_redraft` event. Only
  unverified figures trigger it: section headings are checked by the eval suite.
- **Run usage in `run_finished`.** The usage callback keeps per-thread totals in memory; the
  supervisor's `run_finished` line carries `model_calls`, `input_tokens` and `output_tokens`.
  A run that ends without FINISH drops its tally.
- **RunPolicy.** The six run limits moved from module constants into `RunPolicy` in
  `config.py` (overridable as `RUN_POLICY__...`). The manifest keeps the same keys, and the
  agent version was unchanged by the move (`0.1.0+ac2f6794297e` before and after).

## 2026-10-08 — CI moved to GitHub Actions

The company GitLab runner kept running out of disk while installing the job's dependencies, so
GitLab CI never completed. The job moved to `.github/workflows/ci.yml` and `.gitlab-ci.yml` was
removed. Same steps: Python 3.11, a venv at `./.venv` (where `[tool.pyright]` looks), CPU-only
torch, `.[dev,prod]`, then `scripts/ci_checks.sh`, which `scripts/ci_local.sh` also runs, so a
local pre-push run still matches CI. Runs on every push and pull request, cancels superseded
runs, and uploads the JUnit report. `versioning.git_sha()` now also reads `GITHUB_SHA`.

## 2026-10-08 — Audit close-out: least privilege, token budget, retry from checkpoint

- **MCP least privilege.** The server process gets an allowlisted environment (PATH, HOME,
  locale, temp, venv, `REPORTS_DIR`) and the reports directory as its working directory: its
  settings read `.env` from the working directory, so the project root would have handed it the
  API keys anyway. The unreviewed `run_reporting_pipeline` was removed; every report write goes
  through the approval interrupt.
- **Token budget.** `RunPolicy.max_run_tokens` (50,000; a normal run uses ~14,000; 0 disables).
  At each supervisor decision a code rule compares the run's tokens so far (the in-process tally
  `observability.run_usage_so_far`) with the budget and, when spent, routes straight to
  Reporting with what was gathered; recorded as `policy / token_budget_reached`.
- **Retry from checkpoint.** The error boundary records `failed_node`.
  `graph.retry_failed_run(graph, thread_id)` clears the error, writes the state back "as" the
  step before the failed one (`update_state(..., as_node=...)`, with `next` set when the
  supervisor's edge has to lead back to it), and continues with `invoke(None)`. A failed report
  write re-enters after drafting, so the same draft returns to review. CLI: `--retry`.

### Same day — long-term memory: reviewer feedback across runs

Per-run state lives in the checkpointer; `memory.py` adds what outlives a run. A reviewer's
rejection feedback is stored in the LangGraph store (`SqliteStore` at `data/memory.sqlite`, or
`PostgresStore` with `DATABASE_URL`; one per process like the checkpointer), and each new draft
gets the newest notes from other runs (`RunPolicy.max_reviewer_notes`, 5; 0 turns it off),
fenced as `<reviewer_guidance>` and labelled as preferences, not facts. Notes are sanitized on
the way in -- injection-like notes refused, PII and secrets redacted on the full text and only
then capped, which a test caught being done in the wrong order (a key cut at the limit was too
short to match and its prefix was stored) -- and pruned with the audit log after 90 days.
Production graphs (CLI, Gradio) are compiled with the store; eval graphs without it, so the
release gate doesn't depend on what reviewers wrote.


## 2026-10-08 — Dated corpus and source freshness

The corpus grew from 3 to 12 documents, and documents now carry dates, so two
sources can disagree and one of them is out of date.

- **Header.** Each file in `data/raw/` starts with a `---` block: `entity`
  (a company, `Market`, or `Multiple`), `doc_type` and `as_of` (`YYYY-MM`).
  `ingestion/loaders.py::parse_front_matter` strips it into metadata; a missing
  or malformed field is dropped and the document behaves as undated.
- **Topic and entity per chunk.** `retrieval/evidence.py` gives each chunk a topic
  from its deepest matching heading (pricing, customers, headcount, market_size,
  strengths, weaknesses, recent_moves, else other). In a `Multiple` document a
  vendor sub-heading equal to a profile's entity attributes the chunk to that
  company; anything else is `Market`.
- **Supersession.** A finding is superseded when another finding about the same
  entity and topic has a later `as_of` and states a figure of its own. Undated
  findings, topic `other` and equal dates never supersede. A newer section with no
  figures (the forecast teaser) can't hide real numbers.
- **Where it shows.** Prompts label findings `source · entity · topic · as of
  YYYY-MM (superseded by X)`; the supervisor keeps the bare source in brackets so
  coverage citations still match. Figures only a superseded finding supports get
  ` [outdated]` (`stale_figure`, also a self-check trigger); analytics inputs from
  them raise `stale_tool_input`. A figure the report itself labels as outdated (or
  with the older date) is left alone. A bare number exactly equal to a finding's
  year is read as a date, so "as of 2026-03" isn't flagged; the year is never added
  to the evidence set, which would ground every figure within 1% of it.
- **Evals.** `check_freshness` (an older figure must be labelled outdated/previous
  or with its date) runs on every reporting case; `check_inputs_current` on every
  analytics case. New cases: two retrieval, two reporting conflicts, one analytics,
  one supervisor (the teaser doesn't answer market size), one full pipeline.
- **Corpus test.** `tests/ingestion/test_corpus.py` fails if any figure other than
  Acme's $49 and Globex's $120K/$350K is left stale, so a new document must restate
  every figure of a section it supersedes.

### Reseeded corpus and rerank floor

Reseeding the 12-document corpus gives 66 sections and 70 leaf chunks. Best
cross-encoder score among the top-4 retrieved chunks, per query:

| Query | Kind | Best score | Best source |
|---|---|---|---|
| Acme Starter price per seat | relevant | 9.83 | pricing_benchmark_2026.md |
| Initech pricing plans | relevant | 7.39 | competitor_initech.md |
| Globex annual contract value | relevant | 9.99 | competitor_globex.md |
| embedded analytics trends | relevant | 4.81 | analyst_note_embedded_bi.md |
| BI market size and growth | relevant | 5.88 | market_overview.md |
| Hooli customers | relevant | 6.09 | competitor_hooli.md |
| Vandelay freight KPIs | relevant | 3.68 | competitor_vandelay.md |
| Umbrella viewer pricing | relevant | 8.49 | competitor_umbrella.md |
| best pizza in Naples | off-topic | -11.18 | — |
| how to train a puppy | off-topic | -11.25 | — |
| football world cup winners | off-topic | -10.88 | — |
| symptoms of the flu | off-topic | -11.22 | — |

The gap (relevant ≥ 3.7, off-topic ≤ -10.9) still contains `-8.0`, so
`rerank_score_floor` stays unchanged.

## 2026-10-09 — Structured analytics handoff (audit 2.3)

Analytics' reasoning used to end in a discarded text reply; Reporting saw only raw tool
results. Now each result gets an ID (`r1`, `r2`, …) in its tool reply, and the loop ends with a
`submit_analysis` call carrying up to 5 insights that cite those IDs. It reuses the loop's final
model call, so no call is added. `validate_insights` drops insights with unknown IDs, text over
300 characters, or figures current evidence doesn't support (`ungrounded_insight`). Reporting
renders "Key insights" above "Computed metrics" and is told to build the Analysis section from
them; without insights the prompt is unchanged. The release gate's analytics cases now require
at least one grounded insight (`check_insights`), and a reporting case drafts from insights.
The response cache (off by default) doesn't store insights: a cache hit reports without them.

### Same day — every analytics turn must call a tool

The first gate runs showed gpt-4o-mini ending 4 of 12 analytics runs with a text reply instead of
`submit_analysis`, so a third of reports got no insights. The loop now binds the tools with
`tool_choice="any"` (OpenAI `required`, Anthropic `any`): the model can't end on text, so the only
way to finish is `submit_analysis`. On the last allowed turn the choice is `submit_analysis`
itself, so a run that computes until the cap still hands over a summary (not with a one-turn
budget, which would leave nothing to summarise). No model call is added. The prompt tells the model
to submit an empty list when there is nothing to compute. Both choices are in the version manifest
(`analytics_tool_choice`).
