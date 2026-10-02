from market_research_team.caching import store
from market_research_team.config import settings


def test_put_get_roundtrip(tmp_path):
    conn = store.get_connection(tmp_path / "cache.sqlite")

    store.put(conn, layer="retrieval", key_hash="abc", version="v1", value={"x": 1}, ttl_seconds=60)
    result = store.get(conn, layer="retrieval", key_hash="abc", version="v1")

    assert result == {"x": 1}


def test_get_missing_key_returns_none(tmp_path):
    conn = store.get_connection(tmp_path / "cache.sqlite")

    assert store.get(conn, layer="retrieval", key_hash="missing", version="v1") is None


def test_get_version_mismatch_is_a_miss(tmp_path):
    conn = store.get_connection(tmp_path / "cache.sqlite")
    store.put(conn, layer="retrieval", key_hash="abc", version="v1", value={"x": 1}, ttl_seconds=60)

    assert store.get(conn, layer="retrieval", key_hash="abc", version="v2") is None


def test_get_expired_entry_is_a_miss(tmp_path):
    conn = store.get_connection(tmp_path / "cache.sqlite")
    store.put(
        conn,
        layer="retrieval",
        key_hash="abc",
        version="v1",
        value={"x": 1},
        ttl_seconds=10,
        now=0.0,
    )

    assert store.get(conn, layer="retrieval", key_hash="abc", version="v1", now=100.0) is None


def test_current_vectorstore_version_defaults_to_sentinel_when_unstamped(tmp_path):
    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()

    assert store.current_vectorstore_version(persist_dir) == "0"


def test_current_vectorstore_version_reads_stamp(tmp_path):
    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()
    (persist_dir / ".cache_version").write_text("some-uuid", encoding="utf-8")

    assert store.current_vectorstore_version(persist_dir) == "some-uuid"


def test_put_purges_every_expired_row(tmp_path):
    conn = store.get_connection(tmp_path / "cache.sqlite")
    store.put(
        conn, layer="retrieval", key_hash="old", version="v1", value={}, ttl_seconds=10, now=0.0
    )

    store.put(
        conn, layer="retrieval", key_hash="new", version="v1", value={}, ttl_seconds=10, now=100.0
    )

    keys = [row[0] for row in conn.execute("SELECT key_hash FROM cache_entries")]
    assert keys == ["new"]


def test_cache_version_changes_when_policy_version_is_bumped(tmp_path, monkeypatch):
    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()
    (persist_dir / ".cache_version").write_text("seed-1", encoding="utf-8")

    monkeypatch.setattr(settings, "cache_policy_version", "1")
    before = store.cache_version(persist_dir)
    monkeypatch.setattr(settings, "cache_policy_version", "2")
    after = store.cache_version(persist_dir)

    assert before != after
    assert before.startswith("seed-1")
