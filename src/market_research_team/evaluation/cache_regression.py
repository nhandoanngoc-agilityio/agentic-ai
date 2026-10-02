"""Cold-vs-warm cache regression guard for the golden-dataset eval suite.

Runs `offline_eval.run_all` twice -- once against a cleared cache, once
immediately after -- and flags any case whose `passed` outcome flips from
True to False on the warm run. Caching must only affect latency, never
correctness. Only `evaluate_full_pipeline` meaningfully exercises the
retrieval/rerank caches (the other four categories don't touch the
vectorstore), and it requires the vector store to be seeded first
(`python scripts/seed_vectorstore.py`), same prerequisite `offline_eval.py`
already states.

Makes real LLM calls like the rest of the eval suite -- run on demand, not
as part of the hermetic pytest suite.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from market_research_team.caching import stats
from market_research_team.config import settings
from market_research_team.evaluation.offline_eval import EvalResult, run_all, scoped_eval_run


@dataclass
class CacheRegressionReport:
    cold_results: list[EvalResult]
    warm_results: list[EvalResult]
    regressions: list[tuple[EvalResult, EvalResult]]
    cache_hit_rate: float
    # The warm run's own audit log (when run with `run_dir`); the release gate
    # computes latency/cost/tool metrics from it alone.
    warm_audit_path: Path | None = None


def run_cold_vs_warm(
    provider: str | None = None,
    *,
    judge_llm: BaseChatModel | None = None,
    run_dir: Path | None = None,
) -> CacheRegressionReport:
    """Run the suite cold then warm. With `run_dir`, each pass writes to its own
    scoped directory (`cold/`, `warm/`) so the warm run's metrics are not mixed
    with the cold run's. Only the warm pass is judged: it is the one gated."""

    if settings.cache_db_path.exists():
        settings.cache_db_path.unlink()
    stats.reset()

    def _scope(name: str) -> AbstractContextManager[object]:
        return scoped_eval_run(run_dir / name) if run_dir is not None else nullcontext()

    with _scope("cold"):
        cold = run_all(provider)  # type: ignore[arg-type]
    with _scope("warm"):
        warm = run_all(provider, judge_llm=judge_llm)  # type: ignore[arg-type]

    by_key = {(r.category, r.case_name, r.provider): r for r in cold}
    regressions = [
        (by_key[(r.category, r.case_name, r.provider)], r)
        for r in warm
        if by_key.get((r.category, r.case_name, r.provider))
        and by_key[(r.category, r.case_name, r.provider)].passed
        and not r.passed
    ]
    return CacheRegressionReport(
        cold_results=cold,
        warm_results=warm,
        regressions=regressions,
        cache_hit_rate=stats.hit_rate(),
        warm_audit_path=run_dir / "warm" / "audit.jsonl" if run_dir is not None else None,
    )
