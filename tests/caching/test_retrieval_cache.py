from langchain_core.documents import Document

from market_research_team.caching.retrieval_cache import (
    get_cached_retrieval,
    put_cached_retrieval,
    retrieve_for_queries_cached,
)
from market_research_team.config import settings

_DOCS = [Document(page_content="acme pricing", metadata={"chunk_id": "acme-pricing"})]


def _seed_version(persist_dir, version="v1"):
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / ".cache_version").write_text(version, encoding="utf-8")


def test_cold_miss_then_warm_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    assert (
        get_cached_retrieval(["q"], k=4, embedding_model_name="m", persist_dir=persist_dir) is None
    )

    put_cached_retrieval(["q"], _DOCS, k=4, embedding_model_name="m", persist_dir=persist_dir)
    cached = get_cached_retrieval(["q"], k=4, embedding_model_name="m", persist_dir=persist_dir)

    assert cached is not None
    assert cached[0].metadata["chunk_id"] == "acme-pricing"


def test_version_bump_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir, version="v1")
    put_cached_retrieval(["q"], _DOCS, k=4, embedding_model_name="m", persist_dir=persist_dir)

    _seed_version(persist_dir, version="v2")

    assert (
        get_cached_retrieval(["q"], k=4, embedding_model_name="m", persist_dir=persist_dir) is None
    )


def test_disabled_cache_is_always_a_noop_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", False)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    put_cached_retrieval(["q"], _DOCS, k=4, embedding_model_name="m", persist_dir=persist_dir)

    assert (
        get_cached_retrieval(["q"], k=4, embedding_model_name="m", persist_dir=persist_dir) is None
    )


def test_retrieve_for_queries_cached_populates_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    calls = []

    class _FakeVectorstore:
        pass

    def fake_retrieve(vectorstore, queries, *, k=4):
        calls.append(queries)
        return list(_DOCS)

    monkeypatch.setattr(
        "market_research_team.caching.retrieval_cache.retrieve_for_queries", fake_retrieve
    )

    vectorstore = _FakeVectorstore()
    first = retrieve_for_queries_cached(
        vectorstore, ["q"], k=4, embedding_model_name="m", persist_dir=persist_dir
    )
    second = retrieve_for_queries_cached(
        vectorstore, ["q"], k=4, embedding_model_name="m", persist_dir=persist_dir
    )

    assert len(calls) == 1, "second call should be served from cache, not re-invoke retrieval"
    assert first[0].metadata["chunk_id"] == second[0].metadata["chunk_id"]
