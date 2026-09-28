from langchain_core.documents import Document

from market_research_team.caching.rerank_cache import rerank_cached
from market_research_team.config import settings

_DOCS = [Document(page_content="acme pricing", metadata={"chunk_id": "acme-pricing"})]


class _FakeModel:
    def predict(self, pairs):
        return [1.0 for _ in pairs]


def _seed_version(persist_dir, version="v1"):
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / ".cache_version").write_text(version, encoding="utf-8")


def test_rerank_cached_populates_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    predict_calls = []
    model = _FakeModel()
    original_predict = model.predict

    def counting_predict(pairs):
        predict_calls.append(pairs)
        return original_predict(pairs)

    model.predict = counting_predict

    first = rerank_cached(
        "objective", _DOCS, model, reranker_model_name="m", persist_dir=persist_dir
    )
    second = rerank_cached(
        "objective", _DOCS, model, reranker_model_name="m", persist_dir=persist_dir
    )

    assert len(predict_calls) == 1, "second call should be served from cache"
    assert first == second


def test_rerank_cached_disabled_always_recomputes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_enabled", False)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    model = _FakeModel()
    result = rerank_cached(
        "objective", _DOCS, model, reranker_model_name="m", persist_dir=persist_dir
    )

    assert result[0][0].metadata["chunk_id"] == "acme-pricing"
