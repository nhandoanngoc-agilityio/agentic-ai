# LangSmith Eval Upgrade — Design Spec

**Date:** 2026-09-11
**Status:** Approved for planning
**Sub-project of:** repo-wide practice improvements (eval upgrade is sub-project 1 of 6; see
[Scope](#scope) below for the other five and why they're separate)

## Context

The project has a hermetic `pytest` suite (fake LLMs, no network) and a separate "prompt
evaluation regression suite" (`scripts/run_evals.py` → `src/market_research_team/evaluation/`)
that makes real LLM calls against a golden dataset grounded in the sample documents under
`data/raw/`. That suite currently:

- Defines cases in `evaluation/golden_dataset.py` (5 categories: `query_rewrite`,
  `supervisor_decision`, `analytics`, `reporting`, `full_pipeline`).
- Scores them with hand-written deterministic checks in `evaluation/checks.py` (query count,
  keyword coverage, "does any reported value match a plausible grounded figure", substring
  presence) — no LLM-judge scoring exists today.
- Orchestrates everything in `evaluation/offline_eval.py` (`run_all()`), invoked by
  `scripts/run_evals.py`, which prints a pass/fail report and saves JSON to
  `data/eval_results/`.
- LangSmith tracing is a **dormant** capability: `LANGCHAIN_TRACING_V2` /
  `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` are documented in the README's configuration table
  but nothing in the codebase verifies they work end to end, and there is no LLM-judge scoring
  or LangSmith Dataset/Experiment usage anywhere.
- CI is disabled: `.gitlab-ci.yml` ran `ruff` + `pytest` as of commit `fdde9ff`, was later
  commented out, then emptied (`3d08014`, `97f7b2f`). No automated gate runs on push.

This spec covers wiring LangSmith in as both a tracing tool and an LLM-judge scoring layer on
top of the existing deterministic checks, plus restoring CI.

## Scope

This is one of six independent sub-projects identified while scoping a broader
"improve project practices" request (evals, human-in-the-loop approval, MCP architecture,
security, a streaming UI, and a general best-practices pass). Each gets its own spec → plan →
implementation cycle. Security and best-practices findings are *not* getting a standalone plan —
per user decision, they're folded in as running findings wherever a sub-project's plan touches
that code. This spec is scoped to evaluation + the CI restoration that naturally falls out of
touching `scripts/run_evals.py`.

**Explicitly out of scope for this spec:** human-in-the-loop approval, MCP server changes,
the streaming UI, and any security/best-practices changes outside files this plan already
touches.

## Goals

1. Add LLM-judge scoring (correctness/groundedness/relevance, tailored per category) on top of
   — not instead of — the existing deterministic checks.
2. Make LangSmith the primary framework (not DeepEval): tracing + dataset + evaluate() in one
   tool, consistent with this being a LangChain/LangGraph project.
3. Golden dataset cases become a LangSmith-hosted Dataset (LangSmith mirrors
   `golden_dataset.py`, which remains the single source of truth in-repo).
4. Tracing stays opt-in via existing env vars (`LANGCHAIN_TRACING_V2` etc.) — this spec verifies
   and documents that path rather than forcing tracing on.
5. `scripts/run_evals.py` gains an opt-in `--langsmith` flag that additionally syncs datasets and
   runs a LangSmith experiment; default behavior (no flag) is unchanged.
6. Restore CI (`.gitlab-ci.yml`) to run `ruff` + hermetic `pytest` on every push. The real-API
   golden-dataset eval (with or without `--langsmith`) stays a manual pre-release step, not part
   of CI, since it costs real API calls.

## Non-goals

- No DeepEval integration.
- No change to the hermetic `pytest` suite's fakes/mocks.
- No new `Settings` fields for LangSmith credentials — the `langsmith` SDK and LangChain's
  tracer already read `LANGSMITH_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` /
  `LANGCHAIN_PROJECT` from the environment directly; this spec documents them in `.env.example`
  and the README rather than duplicating them into `config.py`.
- No changes to `data/eval_results/` JSON output format.

## Architecture

```
scripts/run_evals.py
  ├── (existing, always runs) offline_eval.run_all(provider)
  │     └── per category: evaluate_query_rewriter / evaluate_supervisor_decision /
  │         evaluate_analytics / evaluate_reporting / evaluate_full_pipeline
  │             → real agent fn (rewrite_and_expand, decide_next_step, ...)
  │             → checks.py deterministic scoring
  │             → EvalResult, saved to data/eval_results/*.json
  │
  └── (new, only with --langsmith) langsmith_eval.run_langsmith_eval(provider)
        └── per category:
              1. sync_dataset(client, category, cases)   # mirrors golden_dataset.py
              2. langsmith.evaluate(
                     target_<category>,                   # adapts example.inputs → real agent fn call
                     data=dataset_name,
                     evaluators=[deterministic_evaluator_<category>, llm_judge_<category>],
                 )
              3. → LangSmithEvalSummary(category, experiment_url, pass_rate)
        → printed alongside the existing local report
```

Both passes are independent and additive: the existing JSON report is always written; the
LangSmith pass, when requested, only adds console output (pass rates + experiment URLs) and
remote state in LangSmith. Neither pass depends on the other's output.

## New module: `src/market_research_team/evaluation/langsmith_eval.py`

### Data model

```python
from dataclasses import dataclass

@dataclass
class JudgeScore:
    passed: bool
    score: float  # 0.0-1.0
    reasoning: str

@dataclass
class LangSmithEvalSummary:
    category: str
    experiment_url: str
    pass_rate: float  # fraction of cases where BOTH evaluators passed
```

`JudgeScore` is also used as the `with_structured_output` schema (as a Pydantic model —
see below; the dataclass above documents the shape, the actual schema class is
`JudgeScoreSchema(BaseModel)` defined in this module for the structured-output call).

### Dataset sync

```python
def sync_dataset(client: "langsmith.Client", category: str, examples: list[dict]) -> "langsmith.schemas.Dataset":
    """Get-or-create `market-research-team-{category}`, wipe existing examples, recreate from
    the current in-repo cases. LangSmith is always a mirror of golden_dataset.py, never a second
    source of truth — this makes every --langsmith run reflect the latest local cases with no
    drift or manual dataset editing in the LangSmith UI."""
```

Each `examples` entry is `{"inputs": {...}, "outputs": {...}}` — see per-category schemas below.

### Per-category components

Each category gets three functions in `langsmith_eval.py`: `target_<category>`,
`deterministic_evaluator_<category>`, `llm_judge_<category>`. All `target_*` functions take
`llm: BaseChatModel` bound via `functools.partial` before being passed to `langsmith.evaluate()`
(LangSmith's `target` signature is `(inputs: dict) -> dict`, so the LLM is closed over rather
than threaded through inputs).

#### `query_rewrite`

- Dataset example: `inputs={"objective": str}`, `outputs={"min_queries": int, "max_queries": int, "required_any_keywords": list[str]}`
- `target_query_rewrite(inputs) -> {"queries": list[str]}`: calls `rewrite_and_expand(inputs["objective"], llm)`
- `deterministic_evaluator_query_rewrite(run, example)`: calls
  `checks.check_query_count(run.outputs["queries"], min_count=example.outputs["min_queries"], max_count=example.outputs["max_queries"])`
  and `checks.check_keyword_coverage(run.outputs["queries"], example.outputs["required_any_keywords"])`;
  returns `{"key": "deterministic", "score": 1.0 if both pass else 0.0, "comment": "; ".join(details)}`
- `llm_judge_query_rewrite(run, example)`: judge prompt asks whether `run.outputs["queries"]` are
  good, non-redundant sub-queries for `example.inputs["objective"]`; returns
  `{"key": "llm_judge", "score": JudgeScore.score, "comment": JudgeScore.reasoning}`

#### `supervisor_decision`

- Dataset example: `inputs={"research_findings": list[ResearchFinding], "analytics_results": list[AnalyticsResult], "report_path": str | None}`,
  `outputs={"allowed_decisions": list[str]}`
- `target_supervisor_decision(inputs) -> {"decision": str}`: builds an `AgentState` from
  `inputs` (same shape as `evaluate_supervisor_decision` today) and calls
  `decide_next_step(state, llm)`
- `deterministic_evaluator_supervisor_decision(run, example)`: `run.outputs["decision"] in example.outputs["allowed_decisions"]`
- `llm_judge_supervisor_decision(run, example)`: judge prompt asks whether the decision is a
  reasonable next step given the state summary (not just technically allowed, but sensible)

#### `analytics`

- Dataset example: `inputs={"objective": str, "findings": list[ResearchFinding]}`,
  `outputs={"min_tool_calls": int, "plausible_values": list[float], "tolerance": float}`
- `target_analytics(inputs) -> {"results": list[AnalyticsResult]}`: calls
  `run_tool_calling_loop(llm, ANALYTICS_TOOLS, inputs["objective"], inputs["findings"])`
- `deterministic_evaluator_analytics(run, example)`: reuses
  `checks.check_min_length` + `checks.check_any_value_matches`
- `llm_judge_analytics(run, example)`: groundedness judge — given `example.inputs["findings"]`
  and `run.outputs["results"]`, does every reported value trace to the source findings with no
  invented numbers

#### `reporting`

- Dataset example: `inputs={"objective": str, "findings": list[ResearchFinding], "results": list[AnalyticsResult]}`,
  `outputs={"required_sections": list[str], "required_facts": list[str]}`
- `target_reporting(inputs) -> {"report": str}`: calls
  `draft_report(inputs["objective"], inputs["findings"], inputs["results"], llm)`
- `deterministic_evaluator_reporting(run, example)`: reuses `checks.check_contains_all` for
  both sections and facts
- `llm_judge_reporting(run, example)`: groundedness + structure judge — is the report faithful
  to the given findings/results and coherently organized

#### `full_pipeline`

- Dataset example: `inputs={"objective": str}`, `outputs={}` (no fixed reference — this case
  checks the run succeeded and produced a grounded report, not a specific expected value)
- `target_full_pipeline(inputs) -> {"error": str | None, "report_path": str | None, "findings_count": int, "results_count": int}`:
  calls `graph_module.run_graph(initial_state)` (same construction as
  `evaluate_full_pipeline` today) and extracts those fields from `final_state`
- `deterministic_evaluator_full_pipeline(run, example)`: `run.outputs["error"] is None and bool(run.outputs["report_path"])`
- `llm_judge_full_pipeline(run, example)`: reads the file at `run.outputs["report_path"]` and
  judges overall quality against `example.inputs["objective"]` (only judge that reads a file,
  since this is the only category whose real output is a report on disk rather than an in-memory
  value)

### Orchestration

```python
def run_langsmith_eval(provider: Literal["anthropic", "openai"] | None = None) -> list[LangSmithEvalSummary]:
    """Requires LANGSMITH_API_KEY; raises a clear RuntimeError (caught by the CLI, see below)
    if it's unset. Mirrors run_all()'s provider-override-then-restore pattern from offline_eval.py."""
```

Loops over the five categories, calling `sync_dataset` then `langsmith.evaluate(...)` for each,
collecting `LangSmithEvalSummary` entries.

## CLI changes: `scripts/run_evals.py`

- New `--langsmith` flag (`action="store_true"`).
- When set: after the existing `run_all()` + `_print_report()` + `save_results()` flow completes
  unchanged, call `langsmith_eval.run_langsmith_eval(provider)` and print a second report section
  (`=== LangSmith experiments ===`, one line per category with pass rate and experiment URL).
- If `--langsmith` is passed but `LANGSMITH_API_KEY` is unset, print
  `"--langsmith requires LANGSMITH_API_KEY to be set"` to stderr and exit non-zero *before*
  running anything (fail fast, matching `scripts/setup_env.py`'s existing fail-fast philosophy)
  — checked at the top of `main()`, not after the (potentially expensive) local eval run.
- Exit code: non-zero if either the existing local checks fail OR any LangSmith category's
  pass rate is below 1.0 (same all-or-nothing failure philosophy as today's suite).

## Config & dependencies

- Add `"langsmith>=0.6,<1.0"` to `pyproject.toml`'s core `dependencies` (currently only pulled
  in transitively via `langchain-core` — this makes the direct import in `langsmith_eval.py`
  an explicit, pinned dependency rather than an implicit one).
- No new `Settings` fields. Document in `.env.example` and README's configuration table:
  - `LANGSMITH_API_KEY` — required for `--langsmith` and for tracing to actually upload
  - `LANGCHAIN_TRACING_V2=true` — enables tracing for any run (CLI, Studio, evals)
  - `LANGCHAIN_PROJECT` — optional, names the LangSmith project traces/experiments land in

## CI restoration

Restore `.gitlab-ci.yml` to its pre-`3d08014` content:

```yaml
stages:
  - test

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  key: "$CI_COMMIT_REF_SLUG"
  paths:
    - .cache/pip

test:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install --upgrade pip
    - pip install -e ".[dev]"
  script:
    - ruff check src tests scripts
    - pytest -q
```

Hermetic only — no `LANGSMITH_API_KEY`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` needed, no
`scripts/run_evals.py` invocation (with or without `--langsmith`) in CI.

## Error handling

- `sync_dataset` failures (network, auth) propagate as-is from the `langsmith` SDK — no custom
  retry/swallow logic, consistent with the project's guardrails philosophy of failing loudly for
  genuinely exceptional conditions rather than masking them (per-node graph guardrails in
  `guardrails.py` are a different concern: they protect the *graph run*, not this offline
  tooling).
- Missing `LANGSMITH_API_KEY` is the one case explicitly checked and given a friendly message
  (see CLI changes above), because it's an expected, common setup gap — not a bug.
- `llm_judge_*` structured-output calls: if the LLM fails to produce valid structured output,
  let the exception propagate (same "don't swallow real failures" reasoning) — this is a manual
  eval tool run by a developer, not a production path needing a fallback.

## Testing

- `tests/test_langsmith_eval.py` (new, hermetic — no network, no real LangSmith calls):
  - `sync_dataset` against a fake/stub client records the right create/delete calls and example
    payloads for a representative category.
  - Each `deterministic_evaluator_*` produces the same pass/fail as directly calling the
    underlying `checks.py` function with equivalent inputs (regression-proofs the wrapping).
  - Each `llm_judge_*` correctly parses a fake structured-output response (`FakeChatModel`
    returning a canned `JudgeScoreSchema`) into the evaluator's returned dict shape.
  - `run_langsmith_eval` is *not* covered here (it needs real credentials) — same boundary the
    project already draws around `offline_eval.run_all()`.
- Existing `tests/test_eval_offline.py` and `tests/test_eval_checks.py` are unaffected — no
  changes to `offline_eval.py`'s public behavior.
- After restoring CI, confirm locally: `ruff check src tests scripts` and `pytest -q` both pass
  clean (the README already claims 103/103 passing; this becomes enforced, not just claimed).

## Documentation updates

- README's "Prompt evaluation regression suite" section: add a subsection on `--langsmith`,
  what it does, and the required env var.
- README's "Known limitations" section: remove the "CI is currently disabled" bullet once
  restored; remove/adjust the "no automated gate" framing.
- README's configuration table: add `LANGSMITH_API_KEY` / `LANGCHAIN_PROJECT` rows (
  `LANGCHAIN_TRACING_V2` is already listed).
- `.env.example`: add the three LangSmith-related vars, commented out / empty by default
  (matching how `DATABASE_URL` is already handled as optional).

## Success criteria

1. `python scripts/run_evals.py` (no flag) behaves identically to today.
2. `python scripts/run_evals.py --langsmith` (with `LANGSMITH_API_KEY` set) syncs 5 datasets,
   runs 5 LangSmith experiments, and prints pass rates + experiment URLs alongside the existing
   local report.
3. `python scripts/run_evals.py --langsmith` without `LANGSMITH_API_KEY` fails fast with a clear
   message and non-zero exit, before making any LLM calls.
4. New hermetic tests for `langsmith_eval.py` pass with no network access and no API keys.
5. `ruff check src tests scripts` and `pytest -q` both pass clean, enforced by restored CI on
   every push.
