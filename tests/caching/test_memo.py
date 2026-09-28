from market_research_team.caching.memo import cached_cross_encoder, cached_embeddings


def test_cached_embeddings_returns_same_object_for_same_model_name():
    cached_embeddings.cache_clear()
    loader_calls = []

    def fake_loader(model_name):
        loader_calls.append(model_name)
        return object()

    first = cached_embeddings("model-a", _loader=fake_loader)
    second = cached_embeddings("model-a", _loader=fake_loader)

    assert first is second
    assert loader_calls == ["model-a"]


def test_cached_embeddings_returns_different_object_for_different_model_name():
    cached_embeddings.cache_clear()

    def fake_loader(model_name):
        return object()

    first = cached_embeddings("model-a", _loader=fake_loader)
    second = cached_embeddings("model-b", _loader=fake_loader)

    assert first is not second


def test_cached_cross_encoder_returns_same_object_for_same_model_name():
    cached_cross_encoder.cache_clear()
    loader_calls = []

    def fake_loader(model_name):
        loader_calls.append(model_name)
        return object()

    first = cached_cross_encoder("model-a", _loader=fake_loader)
    second = cached_cross_encoder("model-a", _loader=fake_loader)

    assert first is second
    assert loader_calls == ["model-a"]
