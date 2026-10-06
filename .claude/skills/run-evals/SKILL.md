---
name: run-evals
description: Use when running or extending the prompt evaluation suite, after changing a system prompt, or when switching LLM providers or models. Explains prerequisites, flags, results, and cost.
---

# Running the eval suite

**Costs real API calls. Confirm with the user before running.**

Prerequisites:
- `.env` has the key for the chosen provider (check `.env.example` for names; never read `.env` directly).
- Vector store seeded: `python scripts/seed_vectorstore.py` (needed by the full-pipeline case).

Commands:
```bash
python scripts/run_evals.py                   # current LLM_PROVIDER
python scripts/run_evals.py --provider openai
python scripts/run_evals.py --compare          # anthropic and openai side by side
python scripts/run_evals.py --langsmith        # adds LangSmith dataset sync + LLM-judge; needs LANGSMITH_API_KEY
```

Results: exit 0 gate passed, 1 gate failed, 2 config/prerequisite error; JSON report in `data/eval_results/<run id>/results.json`. Categories: query_rewrite, retrieval, supervisor_decision, analytics, tool_selection, reporting, full_pipeline, safety, regression.

Adding a case: append to the matching list in `src/market_research_team/evaluation/golden_dataset.py` (`QueryRewriteCase`, `SupervisorDecisionCase`, `AnalyticsCase`, `ReportingCase`, `FullPipelineCase`, `SafetyCase`; a new safety case also needs a check in `evaluation/safety_eval.py`). Ground expectations in `data/raw/` documents, not invented numbers. Deterministic checks live in `evaluation/checks.py`; the harness is `evaluation/offline_eval.py`.

Interpreting a failure: the `detail` string names the failed check. A grounded-number failure means the model invented a figure; fix the prompt, not the check.

## Release gate

Every run ends with a gate verdict per provider covering task success, quality
(LLM judge pinned in `[judge]` of `gate.toml`, currently `openai/gpt-5.4-mini`, so an OpenAI key
is needed even for Anthropic runs; LangSmith scores with `--langsmith`),
tool accuracy, safety (must be 100%), p95 latency and cost per graph run.
Thresholds, tolerances and prices are in `evals/gate.toml`.

- Day-to-day: `python scripts/run_evals.py` (1 repeat).
- Release check: `python scripts/run_evals.py --repeats 3` (~3x the cost).
- Promote: `python scripts/run_evals.py --repeats 3 --update-baseline`, then commit
  `evals/baseline.json`. Refused (exit 2) if the gate failed or repeats < 3.
- Output: `data/eval_results/<run id>/results.json` (+ per-provider `audit.jsonl`
  and `reports/`). "Changed since baseline" lists manifest components that moved.
- A new model needs a `[prices."<model id>"]` entry or the run stops before spending.

## Regression cases from production

`evals/regressions.jsonl` holds curated production failures (see
`scripts/harvest_failures.py` and `scripts/promote_case.py`). Each non-`must_block`
case adds one full graph run per repeat; `run_evals.py` prints the count before it
starts. A new case is never a baseline "case regression".
