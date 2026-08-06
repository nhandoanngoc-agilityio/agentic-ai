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
    openai_model: str = "gpt-5.6"  # verify against OpenAI's current model list before relying on it
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    recursion_limit: int = 20
    checkpoint_db_path: Path = _PROJECT_ROOT / "data" / "checkpoints.sqlite"
    database_url: str | None = None


settings = Settings()
