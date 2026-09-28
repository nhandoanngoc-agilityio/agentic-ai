"""SQLite-backed cache store shared by the retrieval and rerank caches.

Mirrors `checkpointing/store.py`'s connection idiom, but deliberately does
not cache the connection object at module scope: tests need to point at
different `tmp_path` databases via `monkeypatch.setattr(settings,
"cache_db_path", ...)`, and a process-wide cached connection would leak
state across tests. Opening a SQLite connection per call is cheap.
"""

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    layer TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    version TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (layer, key_hash)
);
"""

_UNSEEDED_VERSION = "0"


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def get(
    conn: sqlite3.Connection,
    *,
    layer: str,
    key_hash: str,
    version: str,
    now: float | None = None,
) -> dict | None:
    """Return the deserialized value on a live hit, else None.

    A row whose stored version no longer matches `version`, or whose TTL
    has expired, is a miss -- and is opportunistically deleted so the table
    doesn't grow unboundedly across repeated reseeds.
    """

    now = time.time() if now is None else now
    row = conn.execute(
        "SELECT version, value, expires_at FROM cache_entries WHERE layer = ? AND key_hash = ?",
        (layer, key_hash),
    ).fetchone()
    if row is None:
        return None

    stored_version, value, expires_at = row
    if stored_version != version or expires_at < now:
        conn.execute(
            "DELETE FROM cache_entries WHERE layer = ? AND key_hash = ?", (layer, key_hash)
        )
        conn.commit()
        return None

    return json.loads(value)


def put(
    conn: sqlite3.Connection,
    *,
    layer: str,
    key_hash: str,
    version: str,
    value: dict,
    ttl_seconds: int,
    now: float | None = None,
) -> None:
    now = time.time() if now is None else now
    conn.execute(
        "INSERT OR REPLACE INTO cache_entries "
        "(layer, key_hash, version, value, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
        (layer, key_hash, version, json.dumps(value), now, now + ttl_seconds),
    )
    conn.commit()


def current_vectorstore_version(persist_dir: Path) -> str:
    """Read the version stamp `ingestion.index_build` writes on every reseed.

    Returns a sentinel when the marker is absent (fresh checkout, not yet
    seeded) so cache lookups degenerate to misses rather than erroring.
    """

    marker = persist_dir / ".cache_version"
    if not marker.exists():
        return _UNSEEDED_VERSION
    return marker.read_text(encoding="utf-8").strip()
