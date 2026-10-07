"""Central configuration: paths and ingestion/embedding defaults."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RunPolicy(BaseModel):
    """The behavioural limits of one run, in one place: the harness boundary.

    Each is a safety bound around the agent's own decisions, and each is in
    the agent version manifest (`versioning.py`). Override one from `.env`
    with a `RUN_POLICY__` prefix, e.g. `RUN_POLICY__MAX_ROUTING_VISITS=10`.
    """

    # Safety net, not the stopping rule: with one targeted search per plan
    # item (at most `max_plan_items`) and Analytics only on new findings, a
    # run needs up to 7 decisions before FINISH (broad research, 4 hand-backs,
    # analytics, report).
    max_routing_visits: int = 8
    # Enough to cover a two-company comparison facet by facet; each item can
    # cost a research pass and a supervisor call, so more mean a slower run.
    max_plan_items: int = 4
    # Model turns in the analytics tool-calling loop (a plain Python loop, not
    # covered by LangGraph's recursion limit).
    max_tool_iterations: int = 4
    # Findings kept across research passes, so repeated hand-backs can't grow
    # the analytics and report prompts without bound.
    max_findings: int = 10
    # Human review rounds before a run that keeps being rejected ends.
    max_review_rounds: int = 3
    # Redrafts the agent makes on its own before a human sees a draft, when
    # the output check finds figures it can't trace to the evidence. One per
    # review round: a second attempt rarely fixes what the first didn't, and
    # each costs a full drafting call.
    max_self_check_redrafts: int = 1


class Settings(BaseSettings):
    # `__` separates nested fields in env names: RUN_POLICY__MAX_PLAN_ITEMS.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    run_policy: RunPolicy = Field(default_factory=RunPolicy)

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
    # Per-request deadline and SDK-level retries (rate limits, 5xx, connection
    # errors, with backoff) for every model call. Retrying the call rather than
    # the node keeps work a node already finished (e.g. earlier tool calls in
    # the analytics loop) from running twice. 2 retries = both SDKs' default.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Deadline for one MCP report write, including spawning the server.
    mcp_write_timeout_seconds: float = 30.0
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
