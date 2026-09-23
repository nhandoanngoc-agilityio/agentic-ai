# Market & Competitor Research Analyst Team

Multi-agent LangGraph demo: a supervisor routes between research (RAG), analytics
(Python stat tools), and reporting (writes markdown via a local MCP filesystem
server) plus a Gradio chat UI (`gradio_app/`). Architecture details: `docs/architecture.md`. Design specs:
`docs/superpowers/specs/`, plans: `docs/superpowers/plans/`.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"                      # or: python scripts/setup_env.py
pytest                                       # hermetic, no API key needed
ruff check src tests scripts gradio_app && ruff format --check src tests scripts gradio_app
python scripts/seed_vectorstore.py           # rebuild Chroma index from data/raw/
python scripts/run_graph_cli.py "<objective>" [--thread-id id]   # REAL LLM CALLS
python scripts/run_gradio.py                 # launches the Gradio UI
langgraph dev --no-browser                   # LangGraph Studio on :2024
python scripts/run_evals.py [--provider openai] [--compare] [--langsmith]   # REAL LLM CALLS
```

## Conventions (enforced by the code, not optional)

- Every graph node is registered in `src/market_research_team/graph.py` through
  `with_error_boundary(...)` from `guardrails.py`. Never `add_node` a bare function.
- Nodes are `def node(state: AgentState) -> dict[str, Any]` and return a partial
  state update. State schema lives in `state.py`; add fields there first.
- `pytest` is hermetic: fake LLMs, no network, no seeded vector store, no API key.
  A test that needs a real model belongs in the eval suite, not in `tests/`.
- Supervisor routing is in `agents/supervisor/router.py` (`decide_next_step`,
  `route_from_supervisor`). New routes need the conditional-edge map in `graph.py`.
- Retrieval code (query rewriting, retriever, reranker) lives in `retrieval/`; the
  research node in `agents/research/` only orchestrates it. Guardrails (input validation,
  output filters, shared regex patterns, audit log) are in `security/`; a node that blocks,
  drops or redacts something returns a `guardrail_events` entry. Tests mirror these areas
  under `tests/<area>/`.
- Config is `config.py` (pydantic-settings). Read `.env.example` for keys; never
  open `.env`.
- Ruff: line length 100, rules E/F/I/UP. A hook auto-formats edited `.py` files.
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
