"""Central configuration: paths and ingestion/embedding defaults."""

from pathlib import Path

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

    query_rewrite_model: str = "claude-sonnet-5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    recursion_limit: int = 20
    checkpoint_db_path: Path = _PROJECT_ROOT / "data" / "checkpoints.sqlite"
    database_url: str | None = None


settings = Settings()
