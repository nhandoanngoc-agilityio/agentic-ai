import re

from market_research_team.caching import keys as keys_module
from market_research_team.caching.keys import rerank_key, response_key, retrieval_key
from market_research_team.config import settings


def test_retrieval_key_is_order_independent_over_queries():
    a = retrieval_key(["foo", "bar"], k=4, embedding_model_name="m")
    b = retrieval_key(["bar", "foo"], k=4, embedding_model_name="m")

    assert a == b


def test_retrieval_key_differs_on_k():
    a = retrieval_key(["foo"], k=4, embedding_model_name="m")
    b = retrieval_key(["foo"], k=5, embedding_model_name="m")

    assert a != b


def test_rerank_key_is_order_independent_over_chunk_ids():
    a = rerank_key("objective", ["c1", "c2"], reranker_model_name="m", top_n=5, score_floor=None)
    b = rerank_key("objective", ["c2", "c1"], reranker_model_name="m", top_n=5, score_floor=None)

    assert a == b


def test_rerank_key_differs_on_score_floor():
    a = rerank_key("objective", ["c1"], reranker_model_name="m", top_n=5, score_floor=None)
    b = rerank_key("objective", ["c1"], reranker_model_name="m", top_n=5, score_floor=-8.0)

    assert a != b


def test_namespace_isolates_every_key(monkeypatch):
    monkeypatch.setattr(settings, "cache_namespace", "tenant-a")
    a = retrieval_key(["foo"], k=4, embedding_model_name="m")
    monkeypatch.setattr(settings, "cache_namespace", "tenant-b")
    b = retrieval_key(["foo"], k=4, embedding_model_name="m")

    assert a != b


def test_response_key_changes_with_llm_provider_or_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "model-a")
    base = response_key("objective")

    monkeypatch.setattr(settings, "openai_model", "model-b")
    other_model = response_key("objective")
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    other_provider = response_key("objective")

    assert len({base, other_model, other_provider}) == 3


def test_response_key_changes_when_injection_patterns_change(monkeypatch):
    before = response_key("objective")
    monkeypatch.setattr(keys_module, "INJECTION_PATTERNS", [("new_rule", re.compile("something"))])

    assert response_key("objective") != before
