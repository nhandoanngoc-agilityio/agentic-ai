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
