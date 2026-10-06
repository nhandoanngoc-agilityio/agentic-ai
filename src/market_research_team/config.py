"""Central configuration: paths and ingestion/embedding defaults."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = _PROJECT_ROOT / "data" / "processed"
    vectorstore_dir: Path = _PROJECT_ROOT / "data" / "vectorstore"
    reports_dir: Path = _PROJECT_ROOT / "reports"
    # Production feedback loop (see feedback/). Candidates hold production text
    # and stay out of git; regressions.jsonl is committed and run by the gate.
    candidates_dir: Path = _PROJECT_ROOT / "data" / "regression_candidates"
    regressions_path: Path = _PROJECT_ROOT / "evals" / "regressions.jsonl"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    section_chunk_size: int = 2000
    leaf_chunk_size: int = 800
    leaf_chunk_overlap: int = 120

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    # Provider API keys. pydantic-settings reads them from the shell environment
    # first, then `.env`; llm.py passes them to the client explicitly because
    # nothing exports `.env` into os.environ. repr=False keeps them out of output.
    anthropic_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4o-mini"  # broadly available; model access varies by account/tier
    # Pinned sampling parameters. None leaves the provider default in place;
    # either way the value is recorded in the agent manifest (versioning.py).
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cross-encoder logit floor for retrieval. Measured on the seeded index
    # (2026-09-18): relevant chunks score +6.6 down to about -5, off-topic
    # queries score a flat -11. -8 drops only the clearly irrelevant tail.
    rerank_score_floor: float = -8.0
    audit_log_path: Path = _PROJECT_ROOT / "data" / "audit.jsonl"
    # Retention (see feedback/retention.py and docs/security.md): audit lines and
    # checkpoint threads older than this are pruned, after failures in them are
    # harvested into regression candidates.
    audit_retention_days: int = 90
    auto_prune_enabled: bool = True
    prune_marker_path: Path = _PROJECT_ROOT / "data" / ".last_prune"

    recursion_limit: int = 20
    checkpoint_db_path: Path = _PROJECT_ROOT / "data" / "checkpoints.sqlite"
    # Credentials use repr=False so printing settings (or a test failure that
    # shows them) never leaks a secret.
    database_url: str | None = Field(default=None, repr=False)

    # Retrieval/rerank/response caching (see caching/). Invalidated automatically
    # on vectorstore reseed via a version stamp; TTLs bound staleness otherwise.
    cache_enabled: bool = True
    cache_db_path: Path = _PROJECT_ROOT / "data" / "cache.sqlite"
    retrieval_cache_ttl_seconds: int = 24 * 3600
    # Bump to invalidate every cache layer at once after a change the keys
    # can't see on their own (e.g. a system prompt edit).
    cache_policy_version: str = "1"
    # Mixed into every cache key so tenants/environments sharing one cache
    # file never read each other's entries.
    cache_namespace: str = "default"

    # Full-pipeline response cache (research_findings + analytics_results for
    # an exact repeated objective). Off by default: market research content
    # is time-sensitive, so this is opt-in rather than on-by-default like the
    # other caches. Reporting/approval is never skipped -- see
    # caching/response_cache.py.
    response_cache_enabled: bool = False
    response_cache_ttl_seconds: int = 24 * 3600

    # Langfuse tracing (see observability.py). Optional: tracing is a no-op
    # unless both keys are set, so hermetic tests and unconfigured local runs
    # never touch the network.
    langfuse_public_key: str | None = Field(default=None, repr=False)
    langfuse_secret_key: str | None = Field(default=None, repr=False)
    langfuse_host: str | None = (
        None  # self-hosted base URL, e.g. https://langfuse.internal.example.com
    )
    langfuse_tracing_environment: str = "development"

    # LangSmith key for `run_evals.py --langsmith`. Read from `.env` here because
    # nothing exports `.env` into os.environ, where the LangSmith client looks;
    # `langsmith_eval.py` copies it there before creating the client.
    # LANGCHAIN_API_KEY is accepted as a fallback (same value, older name).
    langsmith_api_key: str | None = Field(default=None, repr=False)
    langchain_api_key: str | None = Field(default=None, repr=False)


settings = Settings()
