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

Results: non-zero exit on any failure; JSON report in `data/eval_results/<timestamp>.json`. Categories: query_rewrite, supervisor_decision, analytics, reporting, full_pipeline.

Adding a case: append to the matching list in `src/market_research_team/evaluation/golden_dataset.py` (`QueryRewriteCase`, `SupervisorDecisionCase`, `AnalyticsCase`, `ReportingCase`, `FullPipelineCase`). Ground expectations in `data/raw/` documents, not invented numbers. Deterministic checks live in `evaluation/checks.py`; the harness is `evaluation/offline_eval.py`.

Interpreting a failure: the `detail` string names the failed check. A grounded-number failure means the model invented a figure; fix the prompt, not the check.
