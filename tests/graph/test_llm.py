"""Tests for the chat model provider factory."""

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from market_research_team.config import settings
from market_research_team.llm import get_chat_model
from market_research_team.observability import TokenUsageCallbackHandler


def test_get_chat_model_defaults_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_model", "claude-sonnet-5")

    llm = get_chat_model()

    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-sonnet-5"


def test_get_chat_model_switches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "gpt-5.6")

    llm = get_chat_model()

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-5.6"


def test_get_chat_model_binds_the_token_usage_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    llm = get_chat_model()

    # On the model itself, so it survives with_structured_output/bind_tools.
    assert isinstance(llm.callbacks, list)
    assert any(isinstance(cb, TokenUsageCallbackHandler) for cb in llm.callbacks)


def test_get_chat_model_passes_pinned_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_temperature", 0.0)
    monkeypatch.setattr(settings, "llm_max_tokens", 2048)

    llm = get_chat_model()

    assert isinstance(llm, ChatAnthropic)
    assert llm.temperature == 0.0
    assert llm.max_tokens == 2048


def test_get_chat_model_accepts_explicit_provider_and_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")

    llm = get_chat_model(provider="anthropic", model="claude-opus-5")

    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-opus-5"


def test_model_id_for_reads_the_provider_setting(monkeypatch: pytest.MonkeyPatch):
    from market_research_team.llm import model_id_for

    monkeypatch.setattr(settings, "openai_model", "gpt-x")
    monkeypatch.setattr(settings, "anthropic_model", "claude-y")

    assert model_id_for("openai") == "gpt-x"
    assert model_id_for("anthropic") == "claude-y"


def test_get_chat_model_passes_the_anthropic_key_from_settings(monkeypatch: pytest.MonkeyPatch):
    # A key kept only in .env reaches the client (nothing exports .env to os.environ).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-fake-from-dotenv")

    llm = get_chat_model(provider="anthropic", model="claude-opus-5")

    assert isinstance(llm, ChatAnthropic)
    assert llm.anthropic_api_key.get_secret_value() == "sk-ant-fake-from-dotenv"


def test_get_chat_model_passes_the_openai_key_from_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "sk-fake-from-dotenv")

    llm = get_chat_model(provider="openai", model="gpt-x")

    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "sk-fake-from-dotenv"


def test_get_chat_model_falls_back_to_the_environment_without_a_settings_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-from-shell")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    llm = get_chat_model(provider="anthropic", model="claude-opus-5")

    assert isinstance(llm, ChatAnthropic)
    assert llm.anthropic_api_key.get_secret_value() == "sk-ant-fake-from-shell"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_get_chat_model_always_pins_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """A call with no deadline can hang a run; both providers get one."""

    monkeypatch.setattr(settings, "llm_timeout_seconds", 12.5)
    monkeypatch.setattr(settings, "llm_max_retries", 4)

    client: Any = get_chat_model(provider=provider, model="m")

    timeout = client.request_timeout if provider == "openai" else client.default_request_timeout
    assert timeout == 12.5
    assert client.max_retries == 4


def test_usage_is_recorded_through_tool_binding_and_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the callback attached by `.with_config(...)`, only plain `.invoke`
    calls were recorded: `bind_tools` and `with_structured_output` dropped it,
    so the planner, supervisor, rewriter and analytics never counted."""

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import BaseModel

    from market_research_team import observability

    recorded: list[str] = []
    monkeypatch.setattr(
        observability.audit,
        "record",
        lambda event, thread_id, **fields: recorded.append(fields["component"]),
    )

    def _generate(self: Any, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any):
        message = AIMessage(
            content='{"a": 1}',
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)

    class _Schema(BaseModel):
        a: int

    llm = get_chat_model(provider="openai", model="m")
    llm.invoke("x", config={"tags": ["reporting_draft"]})
    llm.bind_tools([_Schema]).invoke("x", config={"tags": ["analytics"]})
    try:
        llm.with_structured_output(_Schema).invoke("x", config={"tags": ["planner"]})
    except Exception:
        pass  # parsing the fake reply may fail; the call itself has been recorded

    assert recorded == ["reporting_draft", "analytics", "planner"]
