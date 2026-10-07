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


def _fake_client(client):
    """Stand-in for the lru_cached `_client`; the autouse fixture calls cache_clear."""

    def factory():
        return client

    factory.cache_clear = lambda: None
    return factory


def test_record_feedback_score_sends_a_session_score(monkeypatch: pytest.MonkeyPatch):
    sent: dict = {}

    class _Client:
        def create_score(self, **kwargs):
            sent.update(kwargs)

    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_client", _fake_client(_Client()))

    observability.record_feedback_score("thread-1", "down", "Compare Acme")

    assert sent == {
        "name": "user_feedback",
        "value": 0.0,
        "session_id": "thread-1",
        "data_type": "NUMERIC",
        "comment": "Compare Acme",
    }


def test_record_feedback_score_is_silent_when_disabled_or_failing(monkeypatch: pytest.MonkeyPatch):
    class _Boom:
        def create_score(self, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr(observability, "tracing_enabled", lambda: False)
    observability.record_feedback_score("thread-1", "up")  # no client touched

    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_client", _fake_client(_Boom()))
    observability.record_feedback_score("thread-1", "up")  # must not raise


@pytest.mark.parametrize(
    ("tags", "component"),
    [
        (["seq:step:1", "supervisor_router"], "supervisor_router"),  # inside a graph run
        (["planner"], "planner"),
        (["seq:step:3", "langsmith:hidden"], "untagged"),
        (None, "untagged"),
    ],
)
def test_component_is_the_call_site_tag_not_a_framework_tag(tags, component) -> None:
    from market_research_team.observability import component_from_tags

    assert component_from_tags(tags) == component
