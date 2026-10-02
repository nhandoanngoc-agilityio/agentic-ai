"""Chat model factory: selects the LLM provider from settings.

Every call site builds its model through `get_chat_model()` rather than
importing a provider class directly, so switching `LLM_PROVIDER` in `.env`
is the only thing needed to move between Anthropic and OpenAI — none of
the pure decision/drafting functions elsewhere need to change, since they
all accept a plain `BaseChatModel`.
"""

from langchain_core.language_models import BaseChatModel

from market_research_team.config import settings
from market_research_team.observability import TokenUsageCallbackHandler


def model_id_for(provider: str) -> str:
    """The configured model id for `provider` -- used for price lookup and judges."""

    return settings.openai_model if provider == "openai" else settings.anthropic_model


def get_chat_model(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Construct the configured chat model. Provider packages are imported
    lazily so only the one actually selected needs to be installed.
    `provider`/`model` override settings, for callers that need a specific
    model regardless of `LLM_PROVIDER` (the eval judge).

    Binds `TokenUsageCallbackHandler` so every call site (query rewriting,
    supervisor routing, analytics tool-calling, report drafting) gets local
    token-usage tracking automatically, without each one needing to pass a
    callback itself.
    """

    # Only pass parameters that are pinned, so unset ones keep the provider default.
    params: dict[str, float | int | str] = {}
    if settings.llm_temperature is not None:
        params["temperature"] = settings.llm_temperature
    if settings.llm_max_tokens is not None:
        params["max_tokens"] = settings.llm_max_tokens

    provider = provider or settings.llm_provider
    model_name = model or model_id_for(provider)
    # Pass the key only when settings has one; otherwise the client falls back
    # to its own environment lookup.
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if settings.openai_api_key:
            params["api_key"] = settings.openai_api_key
        chat_model: BaseChatModel = ChatOpenAI(model=model_name, **params)
    else:
        from langchain_anthropic import ChatAnthropic

        if settings.anthropic_api_key:
            params["api_key"] = settings.anthropic_api_key
        chat_model = ChatAnthropic(model=model_name, **params)

    return chat_model.with_config({"callbacks": [TokenUsageCallbackHandler()]})
