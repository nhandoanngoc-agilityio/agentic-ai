# Market & Competitor Research Analyst Team

Multi-agent LangGraph demo: a supervisor routes between research (RAG), analytics
(Python stat tools), and reporting (writes markdown via a local MCP filesystem
server) plus a Gradio chat UI (`gradio_app/`). Architecture details: `docs/architecture.md`. Design specs:
`docs/superpowers/specs/`, plans: `docs/superpowers/plans/` (local only, git-ignored; never commit them).

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"                      # or: python scripts/setup_env.py
pytest                                       # hermetic, no API key needed
ruff check src tests scripts gradio_app && ruff format --check src tests scripts gradio_app
basedpyright                                 # type check; fails only on errors not in .basedpyright/baseline.json
pytest --cov                                 # coverage report; fails under 93%
scripts/ci_local.sh [--worktree]             # run the GitHub Actions CI job locally on a clean copy before pushing
python scripts/seed_vectorstore.py           # rebuild Chroma index from data/raw/
python scripts/run_graph_cli.py "<objective>" [--thread-id id]   # REAL LLM CALLS
python scripts/run_graph_cli.py --retry <thread-id>                # re-run a failed run's failed step
python scripts/run_gradio.py                 # launches the Gradio UI
langgraph dev --no-browser                   # LangGraph Studio on :2024
python scripts/run_evals.py [--provider openai] [--compare] [--langsmith] [--repeats N] [--update-baseline]   # REAL LLM CALLS
python scripts/show_agent_manifest.py [--json | --diff <file>]  # agent version, no LLM calls
python scripts/harvest_failures.py [--since 7d] [--no-langfuse]   # production failures -> candidates, free
python scripts/promote_case.py list|show|promote|reject <id>      # curate candidates
python scripts/prune_data.py [--days 90] [--apply]                 # retention, dry run by default
```

## Conventions (enforced by the code, not optional)

- Every graph node is registered in `src/market_research_team/graph.py` through
  `with_error_boundary(...)` from `guardrails.py`. Never `add_node` a bare function.
- Nodes are `def node(state: AgentState) -> dict[str, Any]` and return a partial
  state update. State schema lives in `state.py`; add fields there first.
- `pytest` is hermetic: fake LLMs, no network, no seeded vector store, no API key.
  A test that needs a real model belongs in the eval suite, not in `tests/`.
- Supervisor routing is in `agents/supervisor/router.py` (`decide_route`,
  `route_from_supervisor`). New routes need the conditional-edge map in `graph.py`. The
  `planner` node runs before the first supervisor decision; `tests/conftest.py` stubs
  `run_planner` with the one-item fallback plan so graph tests stay offline, and points the
  long-term memory store (`memory.py`) at a temp file. Eval graphs are built without a store.
- A fresh run's input comes from `state.new_run_state()`; a new `AgentState` field must be
  reset there (`tests/graph/test_state.py` fails otherwise).
- Retrieval code (query rewriting, retriever, reranker) lives in `retrieval/`; the
  research node in `agents/research/` only orchestrates it. Guardrails (input validation,
  output filters, shared regex patterns, audit log) are in `security/`; a node that blocks,
  drops or redacts something returns a `guardrail_events` entry. Tests mirror these areas
  under `tests/<area>/`.
- Documents in `data/raw/` start with a front-matter header (`entity`, `doc_type`,
  `as_of`); `tests/ingestion/test_corpus.py` checks it and that only the two built-in
  conflicts leave stale figures. A newer document that covers a vendor's topic must
  restate all of that topic's figures, or the older ones get marked outdated.
- Agent version = `<pyproject version>+<fingerprint>` from `versioning.py`, which hashes
  prompts (`SYSTEM_PROMPT` in each agent), model + params, tool schemas, knowledge/index
  and safety limits. It is stamped on Langfuse traces, audit lines and eval results. A new
  behavioural knob (prompt, limit, pattern list) must be added to the manifest there. Run
  limits live in `RunPolicy` (`config.py`, read as `settings.run_policy`), not as module
  constants.
- Release gate: `evals/gate.toml` (thresholds, tolerances, judge model, prices) and
  `evals/baseline.json` (last approved metrics per provider) are committed. Only
  `run_evals.py --repeats 3 --update-baseline` writes the baseline; committing it is the
  promotion. Exit codes: 0 pass, 1 gate failed, 2 config/prerequisite error.
- Feedback loop (`feedback/`): the harvester turns production failures into candidates in
  `data/regression_candidates/` (git-ignored); a person promotes them into the committed
  `evals/regressions.jsonl` (CLI or Gradio "Regressions" tab), which the release gate runs
  as the `regression` category. Never auto-promote.
- Config is `config.py` (pydantic-settings). Read `.env.example` for keys; never
  open `.env`.
- Ruff: line length 100, rules E/F/I/UP. A hook auto-formats edited `.py` files.
- Types: `basedpyright` in basic mode (`[tool.pyright]`, also read by Pylance). Existing errors
  live in `.basedpyright/baseline.json`; new code must not add any. After fixing old ones, run
  `basedpyright --writebaseline` and commit the smaller baseline. Never regenerate it to hide
  a new error.
- Hermetic MCP tests use the SDK's in-process client session (see
  `tests/mcp/test_mcp_server.py`), not a subprocess.

## Cost rules

- `run_evals.py` and `run_graph_cli.py` spend real API money. Ask before running.
- Keep this context small: delegate searches to `explorer` and test runs to
  `test-runner`. Only use `code-reviewer` / `langgraph-debugger` when asked or
  when a review or a real bug justifies a Fable-class agent.

## Delegation map

| Need | Use |
|---|---|
| Find where something lives, summarise a module | `explorer` agent (Haiku) |
| Run tests or lint and get only failures back | `test-runner` agent (Sonnet) |
| Review a diff before merging | `code-reviewer` agent (Fable) |
| Routing loop, state shape, checkpoint/resume bug | `langgraph-debugger` agent (Fable) |
| Add or change a graph node / route | `langgraph-node` skill |
| Run or extend evals, change a prompt or model | `run-evals` skill |
| Add a tool to the MCP server | `mcp-tool` skill |
| Change chunking, retrieval, or reranking | `rag-tuning` skill |

## Guardrails in place (hooks)

- Bash: `rm -rf` on repo dirs, force pushes, `git clean -f`, and writes to `.env` are blocked.
- Files: reading or editing `.env` and `data/checkpoints.sqlite` is blocked.
- Edits to `.py` files are auto-run through `ruff format` and `ruff check --fix`.
