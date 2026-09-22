import pytest

from market_research_team import observability
from market_research_team.config import settings


@pytest.fixture(autouse=True)
def _clear_client_cache():
    observability._client.cache_clear()
    observability.get_langfuse_handler.cache_clear()
    yield
    observability._client.cache_clear()
    observability.get_langfuse_handler.cache_clear()


def test_tracing_disabled_by_default():
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert observability.tracing_enabled() is False


def test_flush_is_a_noop_when_not_configured():
    # Must not attempt to construct a Langfuse client (no network, no keys).
    observability.flush()


def test_tracing_enabled_requires_both_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", None)
    assert observability.tracing_enabled() is False

    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    assert observability.tracing_enabled() is True


def test_get_langfuse_handler_builds_client_from_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://localhost:3000")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    handler = observability.get_langfuse_handler()

    assert handler is not None
    assert observability.get_langfuse_handler() is handler  # cached
