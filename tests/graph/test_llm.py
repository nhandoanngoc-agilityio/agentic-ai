"""Tests for the chat model provider factory."""

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

    # `get_chat_model` binds a token-usage callback via `.with_config(...)`,
    # so the returned object is a `RunnableBinding` wrapping the real model.
    assert isinstance(llm.bound, ChatAnthropic)
    assert llm.bound.model == "claude-sonnet-5"


def test_get_chat_model_switches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "gpt-5.6")

    llm = get_chat_model()

    assert isinstance(llm.bound, ChatOpenAI)
    assert llm.bound.model_name == "gpt-5.6"


def test_get_chat_model_binds_the_token_usage_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    llm = get_chat_model()

    callbacks = llm.config.get("callbacks", [])
    assert any(isinstance(cb, TokenUsageCallbackHandler) for cb in callbacks)
