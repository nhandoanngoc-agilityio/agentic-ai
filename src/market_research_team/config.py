"""Central configuration: paths and ingestion/embedding defaults."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = _PROJECT_ROOT / "data" / "processed"
    vectorstore_dir: Path = _PROJECT_ROOT / "data" / "vectorstore"
    reports_dir: Path = _PROJECT_ROOT / "reports"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    section_chunk_size: int = 2000
    leaf_chunk_size: int = 800
    leaf_chunk_overlap: int = 120

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"  # broadly available; model access varies by account/tier
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cross-encoder logit floor for retrieval. Measured on the seeded index
    # (2026-09-18): relevant chunks score +6.6 down to about -5, off-topic
    # queries score a flat -11. -8 drops only the clearly irrelevant tail.
    rerank_score_floor: float = -8.0
    audit_log_path: Path = _PROJECT_ROOT / "data" / "audit.jsonl"

    recursion_limit: int = 20
    checkpoint_db_path: Path = _PROJECT_ROOT / "data" / "checkpoints.sqlite"
    database_url: str | None = None

    # Langfuse tracing (see observability.py). Optional: tracing is a no-op
    # unless both keys are set, so hermetic tests and unconfigured local runs
    # never touch the network.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = (
        None  # self-hosted base URL, e.g. https://langfuse.internal.example.com
    )
    langfuse_tracing_environment: str = "development"


settings = Settings()
