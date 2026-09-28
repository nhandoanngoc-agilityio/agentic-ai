import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from market_research_team import observability
from market_research_team.config import settings
from market_research_team.security import audit


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


def _llm_result(usage: dict | None) -> LLMResult:
    message = AIMessage(content="hi", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_token_usage_callback_records_usage_with_component_from_tags(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(settings, "audit_log_path", log)
    handler = observability.TokenUsageCallbackHandler()

    handler.on_llm_end(
        _llm_result({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        run_id=uuid.uuid4(),
        tags=["query_rewriter"],
    )

    summary = audit.token_usage_summary(log)
    assert summary["query_rewriter"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_token_usage_callback_defaults_to_untagged_component(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(settings, "audit_log_path", log)
    handler = observability.TokenUsageCallbackHandler()

    handler.on_llm_end(
        _llm_result({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        run_id=uuid.uuid4(),
        tags=None,
    )

    summary = audit.token_usage_summary(log)
    assert "untagged" in summary


def test_token_usage_callback_never_raises_when_usage_metadata_missing() -> None:
    handler = observability.TokenUsageCallbackHandler()

    handler.on_llm_end(_llm_result(None), run_id=uuid.uuid4(), tags=["query_rewriter"])
