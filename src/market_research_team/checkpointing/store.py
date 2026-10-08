"""Checkpointer factory: SQLite for local/dev persistence, Postgres for production.

Not used by the graph exported for `langgraph dev` / LangGraph Studio —
the dev server manages its own persistence for that entrypoint. This is
for standalone use (scripts, a future API wrapper) via
`graph.build_production_graph()`.

One checkpointer per process: every caller (the app, the CLI, the retention
prune at startup) shares it, and its connection or pool is closed at exit.
Before, each `get_checkpointer()` call opened a fresh connection that was
never closed.
"""

import atexit
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from market_research_team.config import settings

# Gradio serves requests from a thread pool, so Postgres gets a small pool
# rather than one shared connection. Runs are human-paced; a few is plenty.
_POSTGRES_POOL_MIN_SIZE = 1
_POSTGRES_POOL_MAX_SIZE = 4

_lock = threading.Lock()
_current: tuple[str, BaseCheckpointSaver, Callable[[], None]] | None = None
_atexit_registered = False


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the process-wide checkpointer for the current configuration.

    Uses Postgres when `settings.database_url` is set, otherwise a local
    SQLite file. Built on first use and reused after that; if the configured
    target changes (tests point `checkpoint_db_path` at a temp dir), the old
    one is closed and a new one built. The Postgres path lazily imports
    `langgraph-checkpoint-postgres`/`psycopg` (an optional `prod` extra) so
    this module — and the SQLite default — stay usable without them installed.
    """

    global _current, _atexit_registered
    key = settings.database_url or f"sqlite:{settings.checkpoint_db_path}"
    with _lock:
        if _current is not None and _current[0] == key:
            return _current[1]
        _close_current()
        if settings.database_url:
            saver, close = _build_postgres_checkpointer(settings.database_url)
        else:
            saver, close = _build_sqlite_checkpointer(settings.checkpoint_db_path)
        _current = (key, saver, close)
        if not _atexit_registered:
            atexit.register(close_checkpointer)
            _atexit_registered = True
        return saver


def close_checkpointer() -> None:
    """Close the shared checkpointer's connection or pool. Safe to call twice;
    the next `get_checkpointer()` builds a fresh one."""

    with _lock:
        _close_current()


def _close_current() -> None:
    global _current
    if _current is not None:
        _, _, close = _current
        _current = None
        close()


def _build_sqlite_checkpointer(
    db_path: Path,
) -> tuple[BaseCheckpointSaver, Callable[[], None]]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer, conn.close


def _build_postgres_checkpointer(
    database_url: str,
) -> tuple[BaseCheckpointSaver, Callable[[], None]]:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # Same per-connection settings `PostgresSaver.from_conn_string` uses.
    pool = ConnectionPool(
        database_url,
        min_size=_POSTGRES_POOL_MIN_SIZE,
        max_size=_POSTGRES_POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
    except Exception:
        pool.close()
        raise
    return checkpointer, pool.close


# Long-term memory store (see memory.py), one per process like the
# checkpointer, closed at exit.
_current_store: tuple[str, BaseStore, Callable[[], None]] | None = None
_store_atexit_registered = False


def get_memory_store() -> BaseStore:
    """The process-wide long-term memory store: Postgres when
    `settings.database_url` is set, else a local SQLite file."""

    global _current_store, _store_atexit_registered
    key = settings.database_url or f"sqlite:{settings.memory_db_path}"
    with _lock:
        if _current_store is not None and _current_store[0] == key:
            return _current_store[1]
        _close_current_store()
        if settings.database_url:
            store, close = _build_postgres_store(settings.database_url)
        else:
            store, close = _build_sqlite_store(settings.memory_db_path)
        _current_store = (key, store, close)
        if not _store_atexit_registered:
            atexit.register(close_memory_store)
            _store_atexit_registered = True
        return store


def close_memory_store() -> None:
    with _lock:
        _close_current_store()


def _close_current_store() -> None:
    global _current_store
    if _current_store is not None:
        _, _, close = _current_store
        _current_store = None
        close()


def _build_sqlite_store(db_path: Path) -> tuple[BaseStore, Callable[[], None]]:
    from langgraph.store.sqlite import SqliteStore

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store, conn.close


def _build_postgres_store(database_url: str) -> tuple[BaseStore, Callable[[], None]]:
    from langgraph.store.postgres import PostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        database_url,
        min_size=_POSTGRES_POOL_MIN_SIZE,
        max_size=_POSTGRES_POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    try:
        store = PostgresStore(pool)  # type: ignore[arg-type]
        store.setup()
    except Exception:
        pool.close()
        raise
    return store, pool.close
