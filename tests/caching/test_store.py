from market_research_team.caching import store


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
