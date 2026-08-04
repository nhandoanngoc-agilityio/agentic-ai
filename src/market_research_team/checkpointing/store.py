"""Checkpointer factory: SQLite for local/dev persistence, Postgres for production.

Not used by the graph exported for `langgraph dev` / LangGraph Studio —
the dev server manages its own persistence for that entrypoint. This is
for standalone use (scripts, a future API wrapper) via
`graph.build_production_graph()`.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from market_research_team.config import settings


def get_checkpointer() -> BaseCheckpointSaver:
    """Return a ready-to-use checkpointer based on configuration.

    Uses Postgres when `settings.database_url` is set, otherwise falls
    back to a local SQLite file. The Postgres path lazily imports
    `langgraph-checkpoint-postgres`/`psycopg` (an optional `prod` extra)
    so this module — and the SQLite default — stay usable without them
    installed.
    """

    if settings.database_url:
        return _build_postgres_checkpointer(settings.database_url)
    return _build_sqlite_checkpointer(settings.checkpoint_db_path)


def _build_sqlite_checkpointer(db_path: Path) -> BaseCheckpointSaver:
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer


def _build_postgres_checkpointer(database_url: str) -> BaseCheckpointSaver:
    from langgraph.checkpoint.postgres import PostgresSaver

    checkpointer = PostgresSaver.from_conn_string(database_url).__enter__()
    checkpointer.setup()
    return checkpointer
