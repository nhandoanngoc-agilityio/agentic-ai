"""The checkpointer is one per process: shared by every caller, rebuilt when
the configured target changes, and closed (connection or pool) on shutdown.
The Postgres path runs against fake psycopg/langgraph modules -- no database."""

import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from market_research_team.checkpointing import store
from market_research_team.config import settings


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store.close_checkpointer()
    monkeypatch.setattr(settings, "database_url", None)
    monkeypatch.setattr(settings, "checkpoint_db_path", tmp_path / "a.sqlite")
    yield
    store.close_checkpointer()


def test_callers_share_one_checkpointer() -> None:
    assert store.get_checkpointer() is store.get_checkpointer()


def test_changing_the_target_closes_the_old_one_and_builds_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = store.get_checkpointer()
    monkeypatch.setattr(settings, "checkpoint_db_path", tmp_path / "b.sqlite")

    second = store.get_checkpointer()

    assert second is not first
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        first.conn.execute("select 1")


def test_close_checkpointer_closes_the_connection_and_is_idempotent() -> None:
    saver = store.get_checkpointer()

    store.close_checkpointer()
    store.close_checkpointer()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        saver.conn.execute("select 1")
    assert store.get_checkpointer() is not saver


class _FakePool:
    instances: list["_FakePool"] = []

    def __init__(self, conninfo: str, **kwargs: Any) -> None:
        self.conninfo, self.kwargs, self.closed = conninfo, kwargs, False
        _FakePool.instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FakePostgresSaver:
    fail_setup = False

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def setup(self) -> None:
        if self.fail_setup:
            raise RuntimeError("cannot reach database")


@pytest.fixture
def fake_postgres(monkeypatch: pytest.MonkeyPatch):
    _FakePool.instances = []
    _FakePostgresSaver.fail_setup = False
    modules = {
        "psycopg_pool": types.SimpleNamespace(ConnectionPool=_FakePool),
        "psycopg": types.ModuleType("psycopg"),
        "psycopg.rows": types.SimpleNamespace(dict_row="dict_row"),
        "langgraph.checkpoint.postgres": types.SimpleNamespace(PostgresSaver=_FakePostgresSaver),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(settings, "database_url", "postgresql://u:p@db/x")


def test_postgres_uses_a_pool_with_from_conn_string_settings_and_closes_it(
    fake_postgres: None,
) -> None:
    saver = store.get_checkpointer()

    (pool,) = _FakePool.instances
    assert saver.conn is pool
    assert pool.conninfo == "postgresql://u:p@db/x"
    assert pool.kwargs["kwargs"] == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": "dict_row",
    }
    assert pool.kwargs["open"] is True
    assert store.get_checkpointer() is saver  # one pool per process, not per call

    store.close_checkpointer()
    assert pool.closed


def test_postgres_pool_is_closed_when_setup_fails(fake_postgres: None) -> None:
    _FakePostgresSaver.fail_setup = True

    with pytest.raises(RuntimeError, match="cannot reach database"):
        store.get_checkpointer()

    assert _FakePool.instances[0].closed
