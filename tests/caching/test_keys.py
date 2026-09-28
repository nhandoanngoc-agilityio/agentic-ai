from market_research_team.caching.keys import rerank_key, retrieval_key


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
