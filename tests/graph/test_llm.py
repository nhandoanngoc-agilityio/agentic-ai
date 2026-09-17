"""Tests for the chat model provider factory."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from market_research_team.config import settings
from market_research_team.llm import get_chat_model


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
