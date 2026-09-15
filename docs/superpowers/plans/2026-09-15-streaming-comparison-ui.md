# Streaming Comparison UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `ui/` Next.js + CopilotKit app that runs one research objective through both Anthropic
and OpenAI simultaneously against two `langgraph dev` deployments, streams each run live, and once
both sides draft a report, shows a comparison chart (LLM-judge score, run stats, analytics-number
diff) so a human can pick a winner — the winner writes to disk through the existing approval gate,
the loser is discarded.

**Architecture:** Two unmodified `langgraph dev` processes (one per provider) are the only backend
runtime CopilotKit talks to directly (`LangGraphAgent({ deploymentUrl, graphId })` — no custom
Python server needed for streaming or interrupt handling). A small standalone FastAPI service
(`comparison_api.py`) exposes one route, `POST /judge`, reusing the LLM-judge logic already in
`evaluation/langsmith_eval.py`. `reporting_node.py` gets one new interrupt-resume outcome
(`discard`) so a losing candidate ends cleanly instead of looping into another redraft round.

**Tech Stack:** Python 3.11+ (FastAPI, existing LangGraph/LangChain stack), Next.js 15 (App
Router) + TypeScript, `@copilotkit/react-core` v2 + `@copilotkit/runtime`, Recharts, Vitest +
React Testing Library.

**Spec:** `docs/superpowers/specs/2026-09-15-streaming-comparison-ui-design.md`

## Global Constraints

- No changes to `scripts/run_graph_cli.py`'s existing interactive approve/reject behavior — it
  never sends `discard`, so `reporting_node`'s existing branches must behave exactly as they do
  today. Confirmed by `tests/test_reporting_node_interrupt.py`'s 3 existing tests continuing to
  pass unmodified.
- No changes to `evaluation/offline_eval.py`, `scripts/run_evals.py`, or the hermetic `pytest`
  suite's existing fakes/mocks, except the one `langsmith_eval.py` refactor in Task 2 (extraction
  only — same prompt, same evaluator return shape).
- Follow existing Python code style exactly: private `_XxxSchema(BaseModel)` +
  `Field(description=...)` for structured output, `SystemMessage`/`HumanMessage` for prompts,
  every LLM-driven function takes `llm: BaseChatModel` rather than building one internally, no
  `from __future__ import annotations`.
- `ruff check src tests scripts` must stay clean throughout (rules: `E`, `F`, `I`, `UP`,
  line-length 100); the full existing `pytest` suite must stay green throughout.
- New Python deps (`fastapi`, `uvicorn`, `httpx`) go in a new optional `ui` extra in
  `pyproject.toml`, never in core `dependencies` — matches how the `prod` extra keeps the
  Postgres checkpointer optional.
- Frontend: TypeScript strict mode, App Router, no CSS framework (plain CSS — the spec's
  non-goals explicitly rule out styling/design-system scope beyond "clear and functional"),
  Recharts for charts. No hand-typed npm package version numbers in this plan — packages are
  installed via `npm install <pkg>@latest` and pinned by the generated `package-lock.json`. Every
  interactive component file needs a `"use client"` directive (App Router default is server
  components, which can't use React hooks).
- Per the repo-wide policy set earlier: commit this plan and its spec locally as normal, but do
  not `git push` any commit that touches `docs/superpowers/specs/` or `docs/superpowers/plans/`.

---

### Task 1: `reporting_node.py` — discard outcome for the comparison flow

**Files:**
- Modify: `src/market_research_team/agents/reporting/node.py:140-188` (`reporting_node`)
- Test: `tests/test_reporting_node_interrupt.py`

**Interfaces:**
- Consumes: existing `interrupt()` payload shape (`action`, `filename`, `content`, `attempt`,
  `max_attempts`) and the existing resume-decision shape (`{"approved": bool, "feedback": str |
  None}`) — this task adds one new optional key, `discard: bool`, to that resume shape.
- Produces (used by Task 7's frontend wiring, conceptually — no Python consumer): when a resume
  decision is `{"approved": False, "discard": True}`, `reporting_node` returns
  `{"report_path": None, "messages": [...]}` immediately, with no `error` key set and no further
  interrupt.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reporting_node_interrupt.py`:

```python
def test_reporting_node_discards_immediately_without_further_redraft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_calls: list[str | None] = []
    monkeypatch.setattr(
        reporting_node_module, "draft_report", _fake_draft_report_recording(draft_calls)
    )

    graph = _compiled_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "t4"}}

    paused = graph.invoke(_initial_state(), config=config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["attempt"] == 1

    result = graph.invoke(Command(resume={"approved": False, "discard": True}), config=config)

    assert "__interrupt__" not in result
    assert result.get("report_path") is None
    assert result.get("error") is None
    assert draft_calls == [None]  # exactly one draft attempt -- no redraft round was triggered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_reporting_node_interrupt.py -v -k discards_immediately`
Expected: FAIL — after resuming with `discard: True`, the current code falls through to
`feedback = decision.get("feedback")` (which is `None`) and loops into another redraft, so the
graph pauses again on attempt 2 instead of ending; `"__interrupt__" not in result` fails.

- [ ] **Step 3: Implement the discard branch**

In `src/market_research_team/agents/reporting/node.py`, inside `reporting_node`'s loop, insert a
new branch between the existing `if decision.get("approved"):` block and the
`feedback = decision.get("feedback")` line:

```python
        if decision.get("approved"):
            report_path = asyncio.run(write_report_via_mcp(filename, report_markdown))
            return {
                "report_path": report_path,
                "messages": [
                    AIMessage(content=f"Report written to {report_path}.", name="reporting_agent")
                ],
            }
        if decision.get("discard"):
            return {
                "report_path": None,
                "messages": [
                    AIMessage(
                        content="Report discarded (comparison not selected).",
                        name="reporting_agent",
                    )
                ],
            }
        feedback = decision.get("feedback")
```

(Only the new `if decision.get("discard"):` block is new — the `approved` block above it and the
`feedback = ...` line below it are unchanged, shown here only for anchoring.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_reporting_node_interrupt.py -v`
Expected: PASS (4 passed — the 3 existing tests plus the new one).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: same pass count as before this task, plus 1 (all existing tests still pass).

- [ ] **Step 6: Lint**

Run: `ruff check src/market_research_team/agents/reporting/node.py tests/test_reporting_node_interrupt.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/market_research_team/agents/reporting/node.py tests/test_reporting_node_interrupt.py
git commit -m "Add discard outcome to the reporting approval interrupt"
```

---

### Task 2: Extract `judge_report()` from `llm_judge_reporting`

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py:302-320`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `_JudgeScoreSchema` (existing, same file), `SystemMessage`/`HumanMessage` (existing
  imports), `BaseChatModel` (existing import).
- Produces (used by Task 3): `judge_report(objective: str, findings: list[dict[str, Any]],
  results: list[dict[str, Any]], report: str, llm: BaseChatModel) -> _JudgeScoreSchema`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_langsmith_eval.py` (reuses the existing `_FakeStructuredJudgeLLM` class
already defined earlier in this file):

```python
def test_judge_report_returns_structured_score() -> None:
    fake_llm = _FakeStructuredJudgeLLM(score=0.6, passed=True, reasoning="mostly grounded")

    judgment = langsmith_eval.judge_report(
        "Assess Acme vs Globex pricing",
        [{"source": "x", "content": "Acme is $49/seat"}],
        [{"metric": "mean", "value": 49.0, "detail": "d"}],
        "# Report\nAcme is $49/seat.",
        fake_llm,  # type: ignore[arg-type]
    )

    assert judgment.score == 0.6
    assert judgment.passed is True
    assert judgment.reasoning == "mostly grounded"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_langsmith_eval.py -v -k test_judge_report_returns_structured_score`
Expected: FAIL with `AttributeError: module ... has no attribute 'judge_report'`.

- [ ] **Step 3: Extract the function**

In `src/market_research_team/evaluation/langsmith_eval.py`, replace the existing
`llm_judge_reporting` function (lines 309-320) with:

```python
def judge_report(
    objective: str,
    findings: list[dict[str, Any]],
    results: list[dict[str, Any]],
    report: str,
    llm: BaseChatModel,
) -> _JudgeScoreSchema:
    """Groundedness + structure judge over a drafted report. Standalone so both
    llm_judge_reporting (the LangSmith evaluator) and comparison_api.py's live judge
    endpoint (Task 3) share one prompt/logic path instead of duplicating it."""

    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    return structured_llm.invoke(
        [
            SystemMessage(content=_REPORTING_JUDGE_PROMPT),
            HumanMessage(
                content=(
                    f"Objective: {objective}\nFindings: {findings}\nResults: {results}\n"
                    f"Report:\n{report}"
                )
            ),
        ]
    )


def llm_judge_reporting(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    objective = (example.inputs or {}).get("objective", "")
    findings = (example.inputs or {}).get("findings", [])
    results = (example.inputs or {}).get("results", [])
    judgment = judge_report(objective, findings, results, report, llm)
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

This is a pure extraction: `judge_report`'s prompt and message content are exactly what
`llm_judge_reporting` sent before, with one addition — the objective is now included in the judge
context (previously omitted; a pre-existing minor gap noted during sub-project 1's review, now
naturally fixed since `judge_report` needs `objective` as a first-class parameter for the
standalone judge endpoint anyway).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests, including the pre-existing `test_llm_judge_reporting_returns_judge_score`
— it only asserts on the returned dict shape, not exact prompt content, so it's unaffected by the
objective addition).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: same pass count as before this task, plus 1.

- [ ] **Step 6: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "Extract judge_report() as a standalone function"
```

---

### Task 3: `comparison_api.py` — standalone judge FastAPI service

**Files:**
- Modify: `pyproject.toml`
- Create: `src/market_research_team/comparison_api.py`
- Test: `tests/test_comparison_api.py`

**Interfaces:**
- Consumes: `judge_report(objective, findings, results, report, llm) -> _JudgeScoreSchema` (Task
  2), `get_chat_model()` (existing, `llm.py`).
- Produces: `app` (FastAPI instance, exported for `uvicorn
  market_research_team.comparison_api:app`), `POST /judge` route accepting `{objective: str,
  findings: list[dict], results: list[dict], report: str}` and returning `{passed: bool, score:
  float, reasoning: str}`.

- [ ] **Step 1: Add the `ui` extra to `pyproject.toml`**

In `pyproject.toml`, add a new block to `[project.optional-dependencies]`, after the existing
`prod` block and before `dev`:

```toml
ui = [
    "fastapi>=0.115,<1.0",   # standalone judge service backing the streaming comparison UI
    "uvicorn>=0.30,<1.0",    # ASGI server to run it
    "httpx>=0.27,<1.0",      # required by fastapi.testclient.TestClient
]
```

- [ ] **Step 2: Install**

Run: `pip install -e ".[dev,ui]"`
Expected: installs `fastapi`, `uvicorn`, `httpx` with no errors.

- [ ] **Step 3: Write the failing test**

Create `tests/test_comparison_api.py`:

```python
"""Hermetic tests for the standalone judge FastAPI service -- no network, no real LLM calls.

Reuses the same fake structured-output LLM double pattern as test_langsmith_eval.py.
"""

from fastapi.testclient import TestClient

from market_research_team import comparison_api
from market_research_team.evaluation.langsmith_eval import _JudgeScoreSchema


class _FakeStructuredJudgeLLM:
    def __init__(self, score: float, passed: bool, reasoning: str) -> None:
        self._result = _JudgeScoreSchema(passed=passed, score=score, reasoning=reasoning)

    def with_structured_output(self, _schema: object) -> "_FakeStructuredJudgeLLM":
        return self

    def invoke(self, _messages: list[object]) -> _JudgeScoreSchema:
        return self._result


def test_judge_route_returns_score(monkeypatch: object) -> None:
    fake_llm = _FakeStructuredJudgeLLM(score=0.85, passed=True, reasoning="grounded and clear")
    monkeypatch.setattr(comparison_api, "get_chat_model", lambda: fake_llm)
    client = TestClient(comparison_api.app)

    response = client.post(
        "/judge",
        json={
            "objective": "Assess Acme vs Globex pricing",
            "findings": [
                {"source": "x", "content": "Acme is $49/seat", "relevance_score": 0.9}
            ],
            "results": [{"metric": "mean", "value": 49.0, "detail": "d"}],
            "report": "# Report\nAcme is $49/seat.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"passed": True, "score": 0.85, "reasoning": "grounded and clear"}


def test_judge_route_rejects_missing_field() -> None:
    client = TestClient(comparison_api.app)

    response = client.post("/judge", json={"objective": "x", "findings": [], "results": []})

    assert response.status_code == 422  # `report` is required
```

Note: `monkeypatch` is typed `object` above only to keep this snippet import-free; the actual file
must import `pytest` and type it `pytest.MonkeyPatch`, matching the rest of the test suite's style
— see Step 4 for the real import line.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_comparison_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_research_team.comparison_api'`.

- [ ] **Step 5: Fix the test file's monkeypatch typing**

Before implementing, fix the placeholder typing from Step 3: in `tests/test_comparison_api.py`,
add `import pytest` at the top and change `def test_judge_route_returns_score(monkeypatch: object)`
to `def test_judge_route_returns_score(monkeypatch: pytest.MonkeyPatch)`.

- [ ] **Step 6: Implement the service**

Create `src/market_research_team/comparison_api.py`:

```python
"""Standalone FastAPI judge service backing the streaming comparison UI: scores one drafted
report using the same LLM-judge logic as the LangSmith eval suite (evaluation/langsmith_eval.py).

Run via: uvicorn market_research_team.comparison_api:app --port 8001

Uses whatever LLM_PROVIDER this process is started with -- independent of the two providers
being compared in a given run. Prefer starting it with a provider not currently under comparison
to avoid a judge favoring its own provider's output; not enforced in code, since the human
reviewing the comparison UI is always the final decision-maker regardless of the judge score.
"""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from market_research_team.evaluation.langsmith_eval import _JudgeScoreSchema, judge_report
from market_research_team.llm import get_chat_model

app = FastAPI(title="market-research-team comparison judge")


class JudgeRequest(BaseModel):
    objective: str
    findings: list[dict[str, Any]]
    results: list[dict[str, Any]]
    report: str


@app.post("/judge")
def judge(request: JudgeRequest) -> _JudgeScoreSchema:
    llm = get_chat_model()
    return judge_report(request.objective, request.findings, request.results, request.report, llm)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_comparison_api.py -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: same pass count as before this task, plus 2.

- [ ] **Step 9: Lint**

Run: `ruff check src/market_research_team/comparison_api.py tests/test_comparison_api.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/market_research_team/comparison_api.py tests/test_comparison_api.py
git commit -m "Add standalone comparison_api judge service"
```

---

### Task 4: `ui/` scaffold — Next.js app, CopilotKit runtime route, shared types

**Files:**
- Create: `ui/package.json` (generated then hand-edited), `ui/tsconfig.json`,
  `ui/next.config.mjs`, `ui/.gitignore`, `ui/vitest.config.ts`, `ui/vitest.setup.ts`,
  `ui/app/layout.tsx`, `ui/app/globals.css`, `ui/app/page.tsx`, `ui/app/compare/page.tsx`
  (stub, filled in by Task 7), `ui/app/api/copilotkit/route.ts`, `ui/lib/types.ts`,
  `ui/.env.local.example`
- Test: `ui/app/api/copilotkit/route.test.ts`

**Interfaces:**
- Produces (used by Tasks 5-7): `ui/lib/types.ts` exports `ResearchFinding`, `AnalyticsResult`,
  `AgentStateMessage`, `ComparisonAgentState`, `ReportInterruptPayload`, `ResumeDecision` — TS
  mirrors of the Python `AgentState`/`ResearchFinding`/`AnalyticsResult` TypedDicts
  (`src/market_research_team/state.py`) and the reporting interrupt/resume shapes (Task 1).
- Produces: `ui/app/api/copilotkit/route.ts` exports `POST` (Next.js Route Handler convention),
  registering agent ids `"anthropicAgent"` and `"openaiAgent"`, both `graphId:
  "market_research_team"` (matches `langgraph.json`'s key), read from
  `ANTHROPIC_DEPLOYMENT_URL` / `OPENAI_DEPLOYMENT_URL` env vars.

- [ ] **Step 1: Scaffold the npm project**

From the repo root:

```bash
mkdir -p ui
cd ui
npm init -y
npm install next@latest react@latest react-dom@latest @copilotkit/react-core@latest @copilotkit/runtime@latest recharts@latest
npm install -D typescript @types/react@latest @types/react-dom@latest @types/node@latest vitest @testing-library/react @testing-library/jest-dom @vitejs/plugin-react jsdom eslint eslint-config-next
```

Expected: `ui/package.json`, `ui/package-lock.json`, `ui/node_modules/` created with no install
errors.

- [ ] **Step 2: Edit `package.json`'s scripts**

`npm init -y` only creates a bare `package.json` with no build scripts. Open `ui/package.json` and
replace its `"scripts"` block with:

```json
{
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "lint": "eslint .",
  "test": "vitest run"
}
```

- [ ] **Step 3: Write `ui/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 4: Write `ui/next.config.mjs`**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;
```

- [ ] **Step 5: Write `ui/.gitignore`**

```
node_modules/
.next/
.env.local
```

- [ ] **Step 6: Write `ui/vitest.config.ts`**

```typescript
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
  },
});
```

- [ ] **Step 7: Write `ui/vitest.setup.ts`**

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 8: Write `ui/lib/types.ts`**

```typescript
export interface ResearchFinding {
  source: string;
  content: string;
  relevance_score: number;
}

export interface AnalyticsResult {
  metric: string;
  value: number;
  detail: string;
}

export interface AgentStateMessage {
  name?: string | null;
  content: string;
}

/** Mirrors src/market_research_team/state.py's AgentState. */
export interface ComparisonAgentState {
  messages?: AgentStateMessage[];
  objective?: string;
  next?: "research" | "analytics" | "reporting" | "FINISH";
  research_findings?: ResearchFinding[];
  analytics_results?: AnalyticsResult[];
  report_path?: string | null;
  error?: string | null;
}

/** Mirrors the interrupt() payload from reporting_node (node.py). */
export interface ReportInterruptPayload {
  action: "write_report";
  filename: string;
  content: string;
  attempt: number;
  max_attempts: number;
}

/** Mirrors the resume-decision shape reporting_node reads (node.py, Task 1 of this plan). */
export interface ResumeDecision {
  approved: boolean;
  feedback?: string | null;
  discard?: boolean;
}
```

- [ ] **Step 9: Write `ui/app/globals.css`**

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f7f8;
  color: #1a1a1a;
}

.comparison-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  padding: 1.5rem;
}

@media (max-width: 800px) {
  .comparison-columns {
    grid-template-columns: 1fr;
  }
}

.agent-panel {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 1rem;
  background: white;
}

.agent-panel-error {
  color: #b91c1c;
}
```

- [ ] **Step 10: Write `ui/app/layout.tsx`**

```tsx
import "./globals.css";
import "@copilotkit/react-core/v2/styles.css";

import { CopilotKit } from "@copilotkit/react-core/v2";

export const metadata = {
  title: "Market Research Team — Comparison",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit">{children}</CopilotKit>
      </body>
    </html>
  );
}
```

- [ ] **Step 11: Write `ui/app/page.tsx`**

```tsx
import { redirect } from "next/navigation";

export default function HomePage() {
  redirect("/compare");
}
```

- [ ] **Step 12: Write a stub `ui/app/compare/page.tsx`**

Filled in fully by Task 7 — this stub exists so the app builds and the route resolves after this
task:

```tsx
"use client";

export default function ComparePage() {
  return <p style={{ padding: "1.5rem" }}>Comparison UI under construction.</p>;
}
```

- [ ] **Step 13: Write `ui/.env.local.example`**

```
ANTHROPIC_DEPLOYMENT_URL=http://localhost:2024
OPENAI_DEPLOYMENT_URL=http://localhost:2025
NEXT_PUBLIC_JUDGE_API_URL=http://localhost:8001
```

(`NEXT_PUBLIC_` prefix is required for a Next.js env var to be readable from client components —
`ComparisonChart`, in Task 6, calls the judge API directly from the browser.)

- [ ] **Step 14: Write the failing smoke test for the runtime route**

Create `ui/app/api/copilotkit/route.test.ts` — written before `route.ts` exists, so this
genuinely fails first:

```typescript
import { describe, expect, it } from "vitest";
import { POST } from "./route";

describe("copilotkit runtime route", () => {
  it("exports a POST handler", () => {
    expect(typeof POST).toBe("function");
  });
});
```

- [ ] **Step 15: Run the test to verify it fails**

Run: `cd ui && npm run test`
Expected: FAIL — Vitest reports it cannot resolve the import `./route` (the file doesn't exist
yet).

- [ ] **Step 16: Implement `ui/app/api/copilotkit/route.ts`**

```typescript
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphAgent } from "@copilotkit/runtime/langgraph";
import { NextRequest } from "next/server";

const GRAPH_ID = "market_research_team";

const runtime = new CopilotRuntime({
  agents: {
    anthropicAgent: new LangGraphAgent({
      deploymentUrl: process.env.ANTHROPIC_DEPLOYMENT_URL || "http://localhost:2024",
      graphId: GRAPH_ID,
    }),
    openaiAgent: new LangGraphAgent({
      deploymentUrl: process.env.OPENAI_DEPLOYMENT_URL || "http://localhost:2025",
      graphId: GRAPH_ID,
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    endpoint: "/api/copilotkit",
    serviceAdapter: new ExperimentalEmptyAdapter(),
    runtime,
  });

  return handleRequest(req);
};
```

- [ ] **Step 17: Run the test to verify it passes**

Run: `cd ui && npm run test`
Expected: PASS (1 passed).

- [ ] **Step 18: Verify the app builds**

Run: `cd ui && npm run build`
Expected: build succeeds (no TypeScript errors). This is the frontend's equivalent of `ruff check`
— run it after every subsequent task too.

- [ ] **Step 19: Commit**

```bash
cd ..
git add ui/package.json ui/package-lock.json ui/tsconfig.json ui/next.config.mjs ui/.gitignore \
  ui/vitest.config.ts ui/vitest.setup.ts ui/lib/types.ts ui/app/globals.css ui/app/layout.tsx \
  ui/app/page.tsx ui/app/compare/page.tsx ui/app/api/copilotkit/route.ts \
  ui/app/api/copilotkit/route.test.ts ui/.env.local.example
git commit -m "Scaffold ui/ Next.js app with CopilotKit runtime route"
```

(`ui/node_modules/` and `ui/.next/` are excluded by `ui/.gitignore` from Step 5 — confirm with
`git status` before this commit that neither appears in the diff.)

---

### Task 5: `AgentPanel` component

**Files:**
- Create: `ui/components/AgentPanel.tsx`
- Test: `ui/components/AgentPanel.test.tsx`

**Interfaces:**
- Consumes: `ComparisonAgentState`, `ReportInterruptPayload` (Task 4, `ui/lib/types.ts`),
  `useAgent`/`useInterrupt` from `@copilotkit/react-core/v2`.
- Produces (used by Task 7): `AgentPanel` component with props `{ agentId: string; label: string;
  onInterrupt: (payload: ReportInterruptPayload, resolve: (result: string) => void) => void }`.

A note on scope, discovered while writing this plan: the interrupt payload's `content` (the
drafted report) is produced by one blocking `llm.invoke()` call inside `reporting_node` (see
`draft_report` in `node.py`) — LangGraph only publishes a state update at node boundaries, so the
draft text cannot stream token-by-token without changing `draft_report` itself to use `llm.stream()`,
which is out of scope for this spec (not one of its listed components). What genuinely streams
live, and is what this component shows while a run is in progress, is **node-by-node progress**
(`agent.state.messages`, `research_findings`/`analytics_results` counts) — the draft text itself
appears as a single reveal the moment the interrupt fires, not incrementally.

- [ ] **Step 1: Write the failing test**

Create `ui/components/AgentPanel.test.tsx`:

```tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentPanel } from "./AgentPanel";
import type { ReportInterruptPayload } from "@/lib/types";

const mockUseAgent = vi.fn();
const mockUseInterrupt = vi.fn();

vi.mock("@copilotkit/react-core/v2", () => ({
  useAgent: (args: unknown) => mockUseAgent(args),
  useInterrupt: (args: unknown) => mockUseInterrupt(args),
}));

describe("AgentPanel", () => {
  it("renders progress counts from agent state", () => {
    mockUseAgent.mockReturnValue({
      agent: {
        isRunning: true,
        state: {
          messages: [{ name: "analytics_agent", content: "..." }],
          research_findings: [{ source: "a", content: "b", relevance_score: 0.5 }],
          analytics_results: [],
          error: null,
        },
      },
    });
    mockUseInterrupt.mockReturnValue(null);

    render(<AgentPanel agentId="anthropicAgent" label="Anthropic" onInterrupt={vi.fn()} />);

    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText(/analytics_agent/)).toBeInTheDocument();
    expect(screen.getByText(/Research findings: 1/)).toBeInTheDocument();
  });

  it("renders an error when agent state carries one", () => {
    mockUseAgent.mockReturnValue({
      agent: {
        isRunning: false,
        state: {
          messages: [],
          research_findings: [],
          analytics_results: [],
          error: "Recursion limit reached",
        },
      },
    });
    mockUseInterrupt.mockReturnValue(null);

    render(<AgentPanel agentId="openaiAgent" label="OpenAI" onInterrupt={vi.fn()} />);

    expect(screen.getByText(/Recursion limit reached/)).toBeInTheDocument();
  });

  it("calls onInterrupt with the parsed payload and a resolve callback", () => {
    mockUseAgent.mockReturnValue({
      agent: { isRunning: true, state: { messages: [], research_findings: [], analytics_results: [] } },
    });
    const onInterrupt = vi.fn();
    let capturedRender:
      | ((args: { event: { value: ReportInterruptPayload }; resolve: (s: string) => void }) => unknown)
      | undefined;
    mockUseInterrupt.mockImplementation((args: { render: typeof capturedRender }) => {
      capturedRender = args.render;
      return null;
    });

    render(<AgentPanel agentId="anthropicAgent" label="Anthropic" onInterrupt={onInterrupt} />);

    const payload: ReportInterruptPayload = {
      action: "write_report",
      filename: "x.md",
      content: "# Draft",
      attempt: 1,
      max_attempts: 3,
    };
    const resolve = vi.fn();
    capturedRender?.({ event: { value: payload }, resolve });

    expect(onInterrupt).toHaveBeenCalledWith(payload, resolve);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui && npm run test`
Expected: FAIL — `ui/components/AgentPanel.tsx` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `ui/components/AgentPanel.tsx`:

```tsx
"use client";

import { useAgent, useInterrupt } from "@copilotkit/react-core/v2";
import type { ReportInterruptPayload } from "@/lib/types";

export interface AgentPanelProps {
  agentId: string;
  label: string;
  onInterrupt: (payload: ReportInterruptPayload, resolve: (result: string) => void) => void;
}

export function AgentPanel({ agentId, label, onInterrupt }: AgentPanelProps) {
  const { agent } = useAgent({ agentId });

  useInterrupt({
    agentId,
    renderInChat: false,
    render: ({ event, resolve }) => {
      onInterrupt(event.value as ReportInterruptPayload, resolve);
      return null;
    },
  });

  const state = agent.state ?? {};
  const messages = state.messages ?? [];
  const lastNode = messages.length > 0 ? messages[messages.length - 1]?.name ?? "supervisor" : null;
  const findingsCount = (state.research_findings ?? []).length;
  const resultsCount = (state.analytics_results ?? []).length;
  const errorText = state.error;

  return (
    <div className="agent-panel">
      <h2>{label}</h2>
      <p>Status: {agent.isRunning ? (lastNode ?? "starting") : lastNode ? "paused" : "idle"}</p>
      <p>Research findings: {findingsCount}</p>
      <p>Analytics results: {resultsCount}</p>
      {errorText && <p className="agent-panel-error">Error: {errorText}</p>}
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ui && npm run test`
Expected: PASS (all tests, including the 3 new ones).

- [ ] **Step 5: Build check**

Run: `cd ui && npm run build`
Expected: succeeds.

- [ ] **Step 6: Commit**

```bash
git add ui/components/AgentPanel.tsx ui/components/AgentPanel.test.tsx
git commit -m "Add AgentPanel component"
```

---

### Task 6: `ComparisonChart` component

**Files:**
- Create: `ui/components/ComparisonChart.tsx`
- Test: `ui/components/ComparisonChart.test.tsx`

**Interfaces:**
- Consumes: `ResearchFinding`, `AnalyticsResult` (Task 4, `ui/lib/types.ts`), `recharts`
  (`BarChart`, `Bar`, `XAxis`, `YAxis`, `Tooltip`, `ResponsiveContainer`).
- Produces (used by Task 7): `ComparisonChart` component, and its prop type `CandidateData`:

```typescript
export interface CandidateData {
  agentId: string;
  label: string;
  objective: string;
  findings: ResearchFinding[];
  results: AnalyticsResult[];
  report: string;
  startedAtMs: number;
  draftReadyAtMs: number;
}
```

  Props: `{ candidates: [CandidateData, CandidateData]; judgeApiUrl: string; onPick: (index: 0 |
  1) => void }`.

- [ ] **Step 1: Write the failing test**

Create `ui/components/ComparisonChart.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ComparisonChart, type CandidateData } from "./ComparisonChart";

function candidate(overrides: Partial<CandidateData>): CandidateData {
  return {
    agentId: "anthropicAgent",
    label: "Anthropic",
    objective: "Assess Acme vs Globex pricing",
    findings: [{ source: "x", content: "Acme is $49/seat", relevance_score: 0.9 }],
    results: [{ metric: "mean", value: 49, detail: "d" }],
    report: "# Report",
    startedAtMs: 0,
    draftReadyAtMs: 2000,
    ...overrides,
  };
}

describe("ComparisonChart", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ passed: true, score: 0.8, reasoning: "solid" }),
      }),
    );
  });

  it("fetches a judge score per candidate and renders both", async () => {
    const candidates: [CandidateData, CandidateData] = [
      candidate({ agentId: "anthropicAgent", label: "Anthropic" }),
      candidate({ agentId: "openaiAgent", label: "OpenAI" }),
    ];

    render(<ComparisonChart candidates={candidates} judgeApiUrl="http://localhost:8001" onPick={vi.fn()} />);

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(await screen.findAllByText(/0\.8/)).toHaveLength(2);
  });

  it("calls onPick with the chosen candidate's index", async () => {
    const onPick = vi.fn();
    const candidates: [CandidateData, CandidateData] = [
      candidate({ agentId: "anthropicAgent", label: "Anthropic" }),
      candidate({ agentId: "openaiAgent", label: "OpenAI" }),
    ];

    render(<ComparisonChart candidates={candidates} judgeApiUrl="http://localhost:8001" onPick={onPick} />);
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));

    fireEvent.click(screen.getAllByText("Use this one")[1]);

    expect(onPick).toHaveBeenCalledWith(1);
  });

  it("shows 'score unavailable' when the judge request fails, without blocking the pick buttons", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    const candidates: [CandidateData, CandidateData] = [
      candidate({ agentId: "anthropicAgent", label: "Anthropic" }),
      candidate({ agentId: "openaiAgent", label: "OpenAI" }),
    ];

    render(<ComparisonChart candidates={candidates} judgeApiUrl="http://localhost:8001" onPick={vi.fn()} />);

    expect(await screen.findAllByText(/score unavailable/i)).toHaveLength(2);
    expect(screen.getAllByText("Use this one")).toHaveLength(2);
    expect(screen.getAllByText("Use this one")[0]).not.toBeDisabled();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui && npm run test`
Expected: FAIL — `ui/components/ComparisonChart.tsx` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `ui/components/ComparisonChart.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AnalyticsResult, ResearchFinding } from "@/lib/types";

export interface CandidateData {
  agentId: string;
  label: string;
  objective: string;
  findings: ResearchFinding[];
  results: AnalyticsResult[];
  report: string;
  startedAtMs: number;
  draftReadyAtMs: number;
}

interface ComparisonChartProps {
  candidates: [CandidateData, CandidateData];
  judgeApiUrl: string;
  onPick: (index: 0 | 1) => void;
}

interface JudgeScoreState {
  score: number | null;
  reasoning: string | null;
  loading: boolean;
  failed: boolean;
}

async function fetchJudgeScore(judgeApiUrl: string, candidate: CandidateData): Promise<JudgeScoreState> {
  try {
    const response = await fetch(`${judgeApiUrl}/judge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        objective: candidate.objective,
        findings: candidate.findings,
        results: candidate.results,
        report: candidate.report,
      }),
    });
    if (!response.ok) throw new Error(`judge request failed: ${response.status}`);
    const data = (await response.json()) as { score: number; reasoning: string };
    return { score: data.score, reasoning: data.reasoning, loading: false, failed: false };
  } catch {
    return { score: null, reasoning: null, loading: false, failed: true };
  }
}

export function ComparisonChart({ candidates, judgeApiUrl, onPick }: ComparisonChartProps) {
  const [scores, setScores] = useState<[JudgeScoreState, JudgeScoreState]>([
    { score: null, reasoning: null, loading: true, failed: false },
    { score: null, reasoning: null, loading: true, failed: false },
  ]);

  useEffect(() => {
    let cancelled = false;
    Promise.all(candidates.map((candidate) => fetchJudgeScore(judgeApiUrl, candidate))).then(
      ([first, second]) => {
        if (!cancelled) setScores([first, second]);
      },
    );
    return () => {
      cancelled = true;
    };
    // candidates are only ever set once both drafts are ready (see page.tsx), so this
    // effect fires exactly once per comparison round.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates[0].agentId, candidates[1].agentId]);

  const judgeChartData = candidates.map((candidate, index) => ({
    name: candidate.label,
    score: scores[index].score ?? 0,
  }));

  const statsChartData = candidates.map((candidate, index) => ({
    name: candidate.label,
    "tool calls": candidate.results.length,
    "research findings": candidate.findings.length,
    "seconds to draft": Math.round((candidate.draftReadyAtMs - candidate.startedAtMs) / 1000),
  }));

  return (
    <div>
      <h3>Judge score</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={judgeChartData}>
          <XAxis dataKey="name" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Bar dataKey="score" fill="#4f46e5" />
        </BarChart>
      </ResponsiveContainer>

      <h3>Run stats</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={statsChartData}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="tool calls" fill="#0891b2" />
          <Bar dataKey="research findings" fill="#65a30d" />
          <Bar dataKey="seconds to draft" fill="#ea580c" />
        </BarChart>
      </ResponsiveContainer>

      <div className="comparison-columns">
        {candidates.map((candidate, index) => (
          <div key={candidate.agentId} className="agent-panel">
            <h4>{candidate.label}</h4>
            <p>
              {scores[index].failed
                ? "Score unavailable"
                : scores[index].loading
                  ? "Scoring…"
                  : `Judge score: ${scores[index].score} — ${scores[index].reasoning}`}
            </p>
            <p>Analytics: {candidate.results.map((r) => `${r.metric}=${r.value}`).join(", ") || "none"}</p>
            <button type="button" onClick={() => onPick(index as 0 | 1)}>
              Use this one
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ui && npm run test`
Expected: PASS (all tests, including the 3 new ones).

- [ ] **Step 5: Build check**

Run: `cd ui && npm run build`
Expected: succeeds.

- [ ] **Step 6: Commit**

```bash
git add ui/components/ComparisonChart.tsx ui/components/ComparisonChart.test.tsx
git commit -m "Add ComparisonChart component"
```

---

### Task 7: Wire `app/compare/page.tsx` — the full comparison flow

**Files:**
- Modify: `ui/app/compare/page.tsx` (replace the Task 4 stub)
- Test: `ui/app/compare/page.test.tsx`

**Interfaces:**
- Consumes: `AgentPanel` (Task 5), `ComparisonChart` + `CandidateData` (Task 6),
  `ReportInterruptPayload` (Task 4), `useAgent` from `@copilotkit/react-core/v2`.
- Produces: the complete page — objective input, two `AgentPanel`s, a `ComparisonChart` once both
  sides have drafted, and the winner-pick resolve wiring.

- [ ] **Step 1: Write the failing test**

Create `ui/app/compare/page.test.tsx`:

```tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ComparePage from "./page";

const mockUseAgent = vi.fn();
const mockUseInterrupt = vi.fn();
const interruptRenders: Record<string, (args: { event: { value: unknown }; resolve: (s: string) => void }) => unknown> = {};

vi.mock("@copilotkit/react-core/v2", () => ({
  useAgent: (args: { agentId: string }) => mockUseAgent(args),
  useInterrupt: (args: { agentId: string; render: (r: { event: { value: unknown }; resolve: (s: string) => void }) => unknown }) => {
    interruptRenders[args.agentId] = args.render;
    return mockUseInterrupt(args);
  },
}));

vi.mock("@/components/ComparisonChart", () => ({
  ComparisonChart: ({ onPick }: { onPick: (i: 0 | 1) => void }) => (
    <button onClick={() => onPick(0)}>mock-pick-anthropic</button>
  ),
}));

function agentReturn() {
  return { agent: { isRunning: true, state: { messages: [], research_findings: [], analytics_results: [] }, addMessage: vi.fn(), setState: vi.fn(), runAgent: vi.fn() } };
}

describe("ComparePage", () => {
  it("submits an objective and runs both agents", () => {
    mockUseAgent.mockReturnValue(agentReturn());
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run comparison"));

    expect(mockUseAgent).toHaveBeenCalledWith({ agentId: "anthropicAgent" });
    expect(mockUseAgent).toHaveBeenCalledWith({ agentId: "openaiAgent" });
  });

  it("shows the comparison chart and resolves both interrupts on pick", () => {
    mockUseAgent.mockReturnValue(agentReturn());
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run comparison"));

    const anthropicResolve = vi.fn();
    const openaiResolve = vi.fn();
    const payload = { action: "write_report", filename: "x.md", content: "# A", attempt: 1, max_attempts: 3 };
    interruptRenders["anthropicAgent"]({ event: { value: payload }, resolve: anthropicResolve });
    interruptRenders["openaiAgent"]({ event: { value: { ...payload, content: "# B" } }, resolve: openaiResolve });

    expect(screen.getByText("mock-pick-anthropic")).toBeInTheDocument();
    fireEvent.click(screen.getByText("mock-pick-anthropic"));

    expect(anthropicResolve).toHaveBeenCalledWith(JSON.stringify({ approved: true }));
    expect(openaiResolve).toHaveBeenCalledWith(JSON.stringify({ approved: false, discard: true }));
  });

  it("degrades to a single-candidate pick when one side errors before drafting", () => {
    mockUseAgent.mockImplementation((args: { agentId: string }) => {
      if (args.agentId === "openaiAgent") {
        return {
          agent: {
            isRunning: false,
            state: { messages: [], research_findings: [], analytics_results: [], error: "Recursion limit reached" },
            addMessage: vi.fn(),
            setState: vi.fn(),
            runAgent: vi.fn(),
          },
        };
      }
      return agentReturn();
    });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run comparison"));

    const anthropicResolve = vi.fn();
    const payload = { action: "write_report", filename: "x.md", content: "# A", attempt: 1, max_attempts: 3 };
    interruptRenders["anthropicAgent"]({ event: { value: payload }, resolve: anthropicResolve });
    // openaiAgent never fires an interrupt -- it errored before reaching one.

    expect(screen.queryByText("mock-pick-anthropic")).not.toBeInTheDocument(); // no chart, one side errored
    fireEvent.click(screen.getByText("Use this one"));

    expect(anthropicResolve).toHaveBeenCalledWith(JSON.stringify({ approved: true }));
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui && npm run test`
Expected: FAIL — the Task 4 stub page has no objective input, no "Run comparison" button, and
doesn't call `useAgent` for either agent id.

- [ ] **Step 3: Implement**

Replace `ui/app/compare/page.tsx` entirely:

```tsx
"use client";

import { useState } from "react";
import { useAgent } from "@copilotkit/react-core/v2";
import { AgentPanel } from "@/components/AgentPanel";
import { ComparisonChart, type CandidateData } from "@/components/ComparisonChart";
import type { ReportInterruptPayload } from "@/lib/types";

interface PendingInterrupt {
  payload: ReportInterruptPayload;
  resolve: (result: string) => void;
}

const AGENTS = [
  { agentId: "anthropicAgent", label: "Anthropic" },
  { agentId: "openaiAgent", label: "OpenAI" },
] as const;

const JUDGE_API_URL = process.env.NEXT_PUBLIC_JUDGE_API_URL || "http://localhost:8001";

export default function ComparePage() {
  const [objective, setObjective] = useState("");
  const [running, setRunning] = useState(false);
  const [startedAtMs, setStartedAtMs] = useState(0);
  const [interrupts, setInterrupts] = useState<Record<string, PendingInterrupt>>({});

  const anthropicAgent = useAgent({ agentId: "anthropicAgent" }).agent;
  const openaiAgent = useAgent({ agentId: "openaiAgent" }).agent;
  const agentsByI: Record<string, typeof anthropicAgent> = {
    anthropicAgent,
    openaiAgent,
  };

  function handleRun() {
    setInterrupts({});
    setStartedAtMs(Date.now());
    setRunning(true);
    for (const { agentId } of AGENTS) {
      const agent = agentsByI[agentId];
      agent.setState({
        messages: [],
        objective,
        next: "research",
        research_findings: [],
        analytics_results: [],
        report_path: null,
      });
      agent.runAgent();
    }
  }

  function handleInterrupt(agentId: string, payload: ReportInterruptPayload, resolve: (r: string) => void) {
    setInterrupts((prev) => ({ ...prev, [agentId]: { payload, resolve } }));
  }

  function handlePick(index: 0 | 1) {
    const winner = AGENTS[index].agentId;
    const loser = AGENTS[index === 0 ? 1 : 0].agentId;
    interrupts[winner]?.resolve(JSON.stringify({ approved: true }));
    interrupts[loser]?.resolve(JSON.stringify({ approved: false, discard: true }));
    setInterrupts({});
    setRunning(false);
  }

  const draftedAgentIds = AGENTS.filter(({ agentId }) => interrupts[agentId] !== undefined).map(
    ({ agentId }) => agentId,
  );
  const erroredAgentIds = AGENTS.filter(({ agentId }) => agentsByI[agentId].state?.error).map(
    ({ agentId }) => agentId,
  );
  const bothDrafted = draftedAgentIds.length === 2;
  // One side errored before ever reaching the interrupt, and the other side did reach it: the
  // human can only pick the survivor (spec's "degrade to single-candidate mode" error handling).
  const survivorId =
    draftedAgentIds.length === 1 && erroredAgentIds.length === 1 && erroredAgentIds[0] !== draftedAgentIds[0]
      ? draftedAgentIds[0]
      : null;

  function handleUseSurvivor() {
    if (!survivorId) return;
    interrupts[survivorId]?.resolve(JSON.stringify({ approved: true }));
    setInterrupts({});
    setRunning(false);
  }

  const candidates: [CandidateData, CandidateData] | null = bothDrafted
    ? (AGENTS.map(({ agentId, label }) => {
        const agent = agentsByI[agentId];
        const pending = interrupts[agentId];
        return {
          agentId,
          label,
          objective,
          findings: agent.state?.research_findings ?? [],
          results: agent.state?.analytics_results ?? [],
          report: pending.payload.content,
          startedAtMs,
          draftReadyAtMs: Date.now(),
        };
      }) as [CandidateData, CandidateData])
    : null;

  return (
    <main>
      <div style={{ padding: "1.5rem" }}>
        <label htmlFor="objective">Research objective</label>
        <input
          id="objective"
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
          style={{ width: "100%", marginTop: "0.5rem" }}
        />
        <button type="button" onClick={handleRun} disabled={!objective || running} style={{ marginTop: "0.5rem" }}>
          Run comparison
        </button>
      </div>

      <div className="comparison-columns">
        {AGENTS.map(({ agentId, label }) => (
          <AgentPanel
            key={agentId}
            agentId={agentId}
            label={label}
            onInterrupt={(payload, resolve) => handleInterrupt(agentId, payload, resolve)}
          />
        ))}
      </div>

      {candidates && (
        <div style={{ padding: "1.5rem" }}>
          <ComparisonChart candidates={candidates} judgeApiUrl={JUDGE_API_URL} onPick={handlePick} />
        </div>
      )}

      {survivorId && (
        <div style={{ padding: "1.5rem" }}>
          <p>
            {AGENTS.find(({ agentId }) => agentId === erroredAgentIds[0])?.label} failed (
            {agentsByI[erroredAgentIds[0]].state?.error}) — only{" "}
            {AGENTS.find(({ agentId }) => agentId === survivorId)?.label} produced a draft.
          </p>
          <button type="button" onClick={handleUseSurvivor}>
            Use this one
          </button>
        </div>
      )}
    </main>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ui && npm run test`
Expected: PASS (all tests).

- [ ] **Step 5: Build check**

Run: `cd ui && npm run build`
Expected: succeeds.

- [ ] **Step 6: Commit**

```bash
git add ui/app/compare/page.tsx ui/app/compare/page.test.tsx
git commit -m "Wire the comparison page: run both agents, resolve winner/loser interrupts"
```

---

### Task 8: Documentation and manual acceptance check

**Files:**
- Modify: `README.md`
- Create: `ui/README.md`

**Interfaces:** None — this task produces no code, only documentation and a manual verification
pass, matching how sub-project 1 (Task 9) closed out with a documentation-only task.

- [ ] **Step 1: Add a README section**

In the root `README.md`, add a new `## Streaming comparison UI` section after the existing
"Prompt evaluation regression suite" section, with this content:

```markdown
## Streaming comparison UI

A browser UI (`ui/`, Next.js + CopilotKit) that runs one objective through both Anthropic and
OpenAI at once, streams each run live, and — once both sides draft a report — shows a comparison
chart (LLM-judge score, run stats, analytics numbers) so you can pick a winner. The winner writes
to disk through the same human-approval gate every run already goes through; the other candidate
is discarded.

Requires **both** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set — unlike every other flow in this
repo, which needs only one provider's key.

Four processes, each in its own terminal:

```bash
# Terminal 1 — Anthropic-backed graph deployment
LLM_PROVIDER=anthropic langgraph dev --port 2024 --no-browser

# Terminal 2 — OpenAI-backed graph deployment
LLM_PROVIDER=openai langgraph dev --port 2025 --no-browser

# Terminal 3 — judge service (pick a provider not currently under comparison if you can)
pip install -e ".[dev,ui]"
uvicorn market_research_team.comparison_api:app --port 8001

# Terminal 4 — the UI itself
cd ui
cp .env.local.example .env.local   # edit if you changed any of the ports above
npm install
npm run dev
```

Open `http://localhost:3000`, enter an objective, and click "Run comparison". See `ui/README.md`
for frontend-specific details.
```

- [ ] **Step 2: Add a note to README's "Known limitations" section**

In `README.md`'s existing "Known limitations" list, add:

```markdown
- **The streaming comparison UI has no automated browser test coverage.** Its components are
  covered by Vitest/React Testing Library, but the full two-provider run-and-compare flow against
  real `langgraph dev` deployments is a manual check, the same way LangGraph Studio's browser UI
  is.
```

- [ ] **Step 3: Write `ui/README.md`**

```markdown
# Comparison UI

Next.js + CopilotKit frontend for the streaming provider-comparison flow. See the root
`README.md`'s "Streaming comparison UI" section for the full four-process setup — this file
covers only what's specific to this app.

## Setup

```bash
cp .env.local.example .env.local
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Environment variables (`.env.local`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_DEPLOYMENT_URL` | `http://localhost:2024` | Where `LLM_PROVIDER=anthropic langgraph dev` is listening |
| `OPENAI_DEPLOYMENT_URL` | `http://localhost:2025` | Where `LLM_PROVIDER=openai langgraph dev` is listening |
| `NEXT_PUBLIC_JUDGE_API_URL` | `http://localhost:8001` | Where `comparison_api.py`'s `uvicorn` server is listening (public — called from the browser) |

## Scripts

- `npm run dev` — start the Next.js dev server
- `npm run build` — production build (also the fastest way to typecheck the whole app)
- `npm run test` — run the Vitest component test suite
- `npm run lint` — ESLint
```

- [ ] **Step 4: Manual acceptance check**

Not automated — run through this once the plan is fully implemented, with real
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set and the vector store seeded
(`python scripts/seed_vectorstore.py`, if not already done):

1. Start all four processes per the new README section.
2. Open `http://localhost:3000`, submit an objective (e.g. "Assess Acme vs Globex pricing
   strategy").
3. Confirm both panels stream live progress (node names, findings/results counts).
4. Confirm the comparison chart renders once both sides draft, with real judge scores (not
   "Score unavailable" — if it shows that, check `comparison_api`'s terminal for the actual
   error).
5. Click "Use this one" on either candidate. Confirm exactly one new file appears under
   `reports/`, and the other candidate's panel shows "Report discarded (comparison not
   selected)." in its messages with no file written for it.

If any step fails, that's a real bug to fix before considering this plan done — do not mark this
step's checkbox until the check has actually passed.

- [ ] **Step 5: Commit**

```bash
git add README.md ui/README.md
git commit -m "Document the streaming comparison UI"
```
