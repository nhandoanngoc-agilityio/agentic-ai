# Streaming Comparison UI — Design Spec

**Date:** 2026-09-15
**Status:** Approved for planning
**Sub-project of:** repo-wide practice improvements (streaming UI is sub-project 4 of 6; sub-project
1, LangSmith eval upgrade, and sub-project 2, human-in-the-loop approval, are done and merged to
`main`. See [Scope](#scope).)

## Context

The project today has two ways to run the graph: `scripts/run_graph_cli.py` (CLI, interactive
approve/reject prompts for the human-approval gate added in sub-project 2) and `langgraph dev`
(LangGraph Studio, driven so far only via its API, per the README's "Known limitations"). There is
no browser UI, no way to watch a run stream live, and no way to compare providers side by side
outside of `scripts/run_evals.py --compare`'s offline, sequential, no-UI comparison.

`config.py`'s `Settings.llm_provider` is a single process-wide setting read once at import time by
`llm.get_chat_model()` — there is no per-request provider override anywhere in the graph or its
nodes.

`reporting_node.py` (from sub-project 2) already gates the (irreversible) report write behind a
LangGraph `interrupt()`, resumed with `{"approved": bool, "feedback": str | None}`; a rejection
folds `feedback` into the next of up to `_MAX_REVIEW_ROUNDS = 3` redraft attempts.

`evaluation/langsmith_eval.py` already has category-tailored LLM-judge evaluators, including
`llm_judge_reporting(run, example)` — a groundedness/structure judge over a drafted report — but
it's written against LangSmith's `Run`/`Example` objects, not callable as a plain function.

## Scope

This is sub-project 4 of six. Security and general best-practices findings are folded in as running
findings wherever this sub-project's plan touches code, per the standing decision recorded in the
sub-project 1 spec — they are not a separate deliverable here.

This spec covers: a new `ui/` Next.js + CopilotKit app that runs the *same* objective through both
LLM providers side by side, streams each run live, and — once both sides have drafted a report —
shows a comparison chart (LLM-judge score, run stats, analytics-number diff) so a human can pick a
winner. Picking a winner resolves that side's approval interrupt normally (writes to disk); the
other side's interrupt resolves as a new "discard" outcome (ends immediately, nothing written).

**Explicitly out of scope:** MCP server changes (sub-project 3), any change to how a *single*,
non-comparison run behaves (`run_graph_cli.py`'s existing interactive approve/reject flow is
untouched), per-request provider override inside the graph (ruled out in favor of running two
`langgraph dev` processes — see [Architecture](#architecture)), authentication/access control on
the new UI or judge service (local dev tool, not a deployed product), and any styling/design system
beyond "clear and functional."

## Goals

1. Enter one research objective in a browser UI and watch it run through both Anthropic and OpenAI
   simultaneously, with live node-by-node progress and streaming draft text for each.
2. Once both sides reach the human-approval pause, show a comparison: an LLM-judge score per
   candidate, run stats (tool-call count, research visits, wall-clock time) per candidate, and the
   analytics numbers each candidate's report is grounded in, so a human has real signal for picking
   a winner instead of guessing.
3. Picking a winner writes that report to disk exactly as the existing single-run approval flow
   does today; the other candidate is cleanly discarded, not silently left pending or force-rejected
   through the existing 3-round redraft loop.
4. Reuse the existing LLM-judge prompt/logic from `evaluation/langsmith_eval.py` rather than
   duplicating it.
5. No changes to the graph's control flow, node structure, or single-run CLI behavior — the
   comparison UI is purely an additional consumer of the unmodified graph, run twice.

## Non-goals

- No per-request/per-thread provider override mechanism in `config.py`/`llm.py` — two separate
  `langgraph dev` processes (one per provider) is the chosen mechanism; see
  [Architecture](#architecture) for why.
- No new Python web framework beyond one small FastAPI service for judge scoring — everything else
  the UI needs (progress streaming, interrupt handling, state) comes from CopilotKit talking
  directly to the two `langgraph dev` deployments.
- No authentication, deployment config, or production hosting for `ui/` — this is a local
  development/demo tool, matching how `langgraph dev` itself is documented today.
- No automated end-to-end browser test suite — see [Testing](#testing) for what is and isn't
  automated.
- No changes to `scripts/run_evals.py --compare` or the offline eval harness.

## Architecture

Four local processes, none of which is new infrastructure beyond what's already in the repo except
the judge service:

```
┌─────────────────────────┐   ┌─────────────────────────┐
│ langgraph dev --port 2024│   │ langgraph dev --port 2025│
│ LLM_PROVIDER=anthropic   │   │ LLM_PROVIDER=openai      │
│ (unmodified graph)       │   │ (unmodified graph)       │
└────────────┬─────────────┘   └────────────┬─────────────┘
             │ streamed state + interrupt/resume (native LangGraph API)
             ▼                               ▼
        ┌─────────────────────────────────────────┐
        │  ui/ (Next.js + CopilotKit)              │
        │  app/api/copilotkit/route.ts             │
        │    CopilotRuntime({ agents: {            │
        │      anthropicAgent: LangGraphAgent(:2024)│
        │      openaiAgent:    LangGraphAgent(:2025)│
        │    }})                                    │
        │  app/compare/page.tsx                    │
        │    AgentPanel × 2 (useAgent, useLangGraphInterrupt)
        │    ComparisonChart (after both drafted)  │
        └───────────────────┬───────────────────────┘
                             │ POST /judge  {objective, findings, results, report}
                             ▼
        ┌─────────────────────────────────────────┐
        │ uvicorn comparison_api:app --port 8001   │
        │ (any LLM_PROVIDER — the judge, independent│
        │  of the two providers being compared)     │
        └─────────────────────────────────────────┘
```

**Why two `langgraph dev` processes instead of a per-request provider override:** confirmed via
CopilotKit's own LangGraph integration docs that `LangGraphAgent({ deploymentUrl, graphId })`
connects directly to a running deployment — no `copilotkit` Python package, no
`CopilotKitMiddleware`, no custom server wrapper needed for a plain `StateGraph`. Threading a
provider override through `RunnableConfig.configurable` would touch every node that calls
`get_chat_model()` (`supervisor/router.py`, `agents/research/*`, `agents/analytics/node.py`,
`agents/reporting/node.py`) for a single-server convenience that isn't needed — running the
unmodified graph twice, once per provider, is strictly less code and stays consistent with how
`scripts/run_evals.py --compare` already treats "run under provider X" as a whole-process setting.

**Why `useLangGraphInterrupt` needs no protocol change:** it renders whatever JSON payload
`reporting_node`'s `interrupt(...)` call sends and resolves by sending back arbitrary JSON — our
existing `{"action": "write_report", "filename", "content", "attempt", "max_attempts"}` payload and
`{"approved": bool, "feedback": str | None}` resume contract both work unchanged. The only backend
change is a new field on the resume payload (see below).

## Components

### `reporting_node.py` (modified)

Add a `discard` outcome, distinct from the existing "rejected after `_MAX_REVIEW_ROUNDS` rounds"
failure path:

```python
decision = interrupt({...})  # unchanged payload
if decision.get("approved"):
    ...  # unchanged
if decision.get("discard"):
    return {
        "report_path": None,
        "messages": [
            AIMessage(content="Report discarded (comparison not selected).", name="reporting_agent")
        ],
    }
feedback = decision.get("feedback")
```

No `error` is set — a discard is a deliberate, successful outcome of the comparison flow, not a
failure. This only changes behavior when `decision.get("discard")` is truthy; the existing
approve/reject-with-feedback loop is otherwise untouched, so `run_graph_cli.py`'s interactive flow
(which never sends `discard`) behaves exactly as before.

### `evaluation/langsmith_eval.py` (modified — extraction, not new logic)

Extract the judging call currently inline in `llm_judge_reporting(run, example)` into a standalone,
LangSmith-independent function:

```python
def judge_report(
    objective: str,
    findings: list[ResearchFinding],
    results: list[AnalyticsResult],
    report: str,
    llm: BaseChatModel | None = None,
) -> _JudgeScoreSchema:
    """Groundedness + structure judge over a drafted report. Standalone so both the LangSmith
    evaluator below and comparison_api.py's live judge endpoint share one prompt/logic path."""
```

`llm_judge_reporting(run, example)` becomes a thin wrapper: pulls `objective`/`findings`/`results`
from `example.inputs`, `report` from `run.outputs["report"]`, calls `judge_report(...)`, and shapes
the `{"key": "llm_judge", "score": ..., "comment": ...}` dict LangSmith expects. No behavior change
to the existing `--langsmith` eval path — same prompt, same schema, same scoring.

### `src/market_research_team/comparison_api.py` (new)

A single-route FastAPI app:

```python
class JudgeRequest(BaseModel):
    objective: str
    findings: list[ResearchFinding]
    results: list[AnalyticsResult]
    report: str

app = FastAPI()

@app.post("/judge")
def judge(req: JudgeRequest) -> _JudgeScoreSchema:
    return judge_report(req.objective, req.findings, req.results, req.report, get_chat_model())
```

Run via `uvicorn market_research_team.comparison_api:app --port 8001`. Uses whatever
`LLM_PROVIDER` its own process is started with — independent of the two providers being compared
(documented recommendation: prefer a provider not currently under comparison, not enforced in
code, since this is a local dev tool and the human is always the final decision-maker regardless of
the judge score).

### `ui/` (new Next.js app)

- `app/layout.tsx` — wraps children in `<CopilotKit runtimeUrl="/api/copilotkit">`.
- `app/api/copilotkit/route.ts` — `CopilotRuntime({ agents: { anthropicAgent: new
  LangGraphAgent({ deploymentUrl: "http://localhost:2024", graphId: "market_research_team" }),
  openaiAgent: new LangGraphAgent({ deploymentUrl: "http://localhost:2025", graphId:
  "market_research_team" }) } })`.
- `app/compare/page.tsx` — objective input + "Run comparison" button; on submit, calls
  `agent.addMessage(...)` + `agent.runAgent()` on both `useAgent("anthropicAgent")` and
  `useAgent("openaiAgent")` instances.
- `components/AgentPanel.tsx` — one per provider. Renders: current node (derived from the latest
  `messages` entry's `name` field — `supervisor` / `research_agent` / `analytics_agent` /
  `reporting_agent`, matching the `name=` values each node already sets), running counts
  (`research_findings.length`, `analytics_results.length`), the streaming draft text once
  available, and `useLangGraphInterrupt` for that agent's approval payload. Tracks its own
  wall-clock start time (first state event received) and draft-ready time (interrupt received) for
  the run-stats chart — client-side only, no backend change needed.
- `components/ComparisonChart.tsx` — mounts once both panels have an open interrupt. Fetches
  `POST http://localhost:8001/judge` once per candidate (objective/findings/results from that
  panel's agent state, report from the interrupt payload's `content`), and renders three
  comparison views: judge score (bar), run stats (tool calls / research visits / elapsed time),
  and the analytics numbers each side computed (a simple side-by-side list/table, since these are
  category-labeled metrics rather than a single comparable series — a bar chart per shared metric
  name where both sides computed the same one, otherwise listed separately).
- Winner selection: a "Use this one" button per panel, enabled once the chart has rendered. Calls
  `resolve()` on the chosen panel's `useLangGraphInterrupt` with `{"approved": true}` and on the
  other panel's with `{"approved": false, "discard": true}`.

## Data flow

1. User submits an objective on `app/compare/page.tsx`.
2. Both `useAgent` instances fire `runAgent()` against their own `langgraph dev` deployment.
3. Each streams state live; each `AgentPanel` renders node/progress/draft text independently as
   events arrive — no polling.
4. Each run reaches `reporting_node`'s `interrupt()`; `useLangGraphInterrupt` fires per panel.
5. Once *both* panels have an open interrupt, `ComparisonChart` fetches judge scores for both and
   renders alongside the client-computed run stats and analytics diff.
6. User picks a winner. Winner's interrupt resolves `{"approved": true}` → existing write-to-disk
   path runs unchanged, `report_path` streams into that panel's state, UI shows the written path.
   Loser's interrupt resolves `{"approved": false, "discard": true}` → `reporting_node` returns
   immediately per the new branch above; that panel shows "discarded."

## Error handling

- One side's run sets `state["error"]` before reaching the interrupt (e.g. a node's error boundary
  caught a failure): that panel shows the error and stops streaming; `ComparisonChart` degrades to
  single-candidate mode — the surviving panel's "Use this one" becomes the only enabled action, no
  chart, since there's nothing to compare.
- `/judge` request fails (network, LLM error): that side's chart section shows "score unavailable"
  rather than blocking; run stats and analytics numbers (computed client-side, no network
  dependency on `comparison_api`) still render. The human can still pick a winner without a judge
  score — the judge is advisory input, never a gate, consistent with how the human is already the
  final authority in the approval flow from sub-project 2.
- Either `langgraph dev` deployment unreachable: CopilotKit's own connection-error state surfaces
  in that panel (no custom retry/reconnect logic added).

## Testing

- **Python (hermetic, mirrors existing patterns):**
  - `tests/test_reporting_node.py` / `test_reporting_node_interrupt.py`: new case(s) for the
    `discard` branch — resumes with `{"approved": False, "discard": True}`, asserts
    `report_path is None`, no `error` set, no further interrupt raised (loop exits immediately
    rather than redrafting).
  - `tests/test_langsmith_eval.py`: new case(s) for standalone `judge_report()` using the existing
    `FakeChatModel` pattern; existing `llm_judge_reporting` tests updated only if the wrapper's
    call shape changes (it shouldn't — same inputs, same output dict).
  - New `tests/test_comparison_api.py`: FastAPI `TestClient` against `/judge`, with
    `get_chat_model` swapped for a fake via the same override pattern used elsewhere, asserting the
    request/response shape round-trips through `judge_report()`.
- **Frontend (new territory — no JS tooling exists in this repo yet):** Vitest + React Testing
  Library, component-level only — `AgentPanel` and `ComparisonChart` tested as pure
  props-in/rendered-output-out (feeding them canned agent-state/interrupt/judge-response fixtures),
  not integration tests against real `langgraph dev` processes.
- **Manual acceptance check** (not automated, documented in the plan and README): start all four
  processes with real `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, run one objective through the
  comparison UI end to end, confirm both panels stream live, the chart renders with real judge
  scores, picking a winner writes exactly one file to `reports/`, and the other candidate shows
  "discarded" with nothing written. This mirrors how the README already treats LangGraph Studio's
  browser UI as manually verified rather than covered by `pytest`.

## Config & dependencies

- `pyproject.toml`: new optional extra `ui` = `["fastapi>=0.115", "uvicorn>=0.30"]` — not added to
  core `dependencies`, matching how the `prod` extra keeps the Postgres checkpointer optional today.
  Installed via `pip install -e ".[dev,ui]"`.
- `ui/`: new `package.json` (Next.js, `@copilotkit/react-core`, `@copilotkit/runtime`, a charting
  library — Recharts, to match common CopilotKit example usage), `ui/.env.local.example` with
  `ANTHROPIC_DEPLOYMENT_URL=http://localhost:2024`, `OPENAI_DEPLOYMENT_URL=http://localhost:2025`,
  `JUDGE_API_URL=http://localhost:8001`.
- No changes to `langgraph.json`, `config.py`, or any existing `.env` variable — the two
  `langgraph dev` processes are distinguished purely by the `LLM_PROVIDER` env var each is started
  with, exactly as a single-provider run is configured today.

## Documentation updates

- README: new "Streaming comparison UI" section — what it does, the four processes and the exact
  commands to start each, required env vars (`ANTHROPIC_API_KEY` **and** `OPENAI_API_KEY` both
  needed simultaneously, unlike every other flow in this repo which needs only one), and a link to
  `ui/README.md` for frontend-specific setup.
- `ui/README.md` (new): Next.js app setup (`npm install`, `npm run dev`), the three env vars above.
- Root README's "Known limitations" section: add a note that the comparison UI has no automated
  browser test coverage, same framing already used for LangGraph Studio's UI.

## Success criteria

1. `POST /judge` returns a valid `_JudgeScoreSchema` for a hand-constructed request with no network
   access beyond the configured LLM provider.
2. New hermetic pytest cases (discard branch, `judge_report()`, `/judge` route) pass with no API
   keys, alongside the full existing suite (`ruff check src tests scripts`, `pytest -q` both clean).
3. With both `langgraph dev` processes and `comparison_api` running locally and real API keys set,
   submitting one objective in `ui/` streams both panels live, renders a chart once both drafts are
   ready, and picking a winner writes exactly one file to `reports/` while the other candidate is
   cleanly discarded (no file, no error, no lingering interrupt).
4. `scripts/run_graph_cli.py`'s existing single-run interactive approve/reject flow is unchanged —
   confirmed by the existing `test_reporting_node_interrupt.py` cases continuing to pass unmodified.
