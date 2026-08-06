"""Chat model factory: selects the LLM provider from settings.

Every call site builds its model through `get_chat_model()` rather than
importing a provider class directly, so switching `LLM_PROVIDER` in `.env`
is the only thing needed to move between Anthropic and OpenAI — none of
the pure decision/drafting functions elsewhere need to change, since they
all accept a plain `BaseChatModel`.
"""

from langchain_core.language_models import BaseChatModel

from market_research_team.config import settings


def get_chat_model() -> BaseChatModel:
    """Construct the configured chat model. Provider packages are imported
    lazily so only the one actually selected needs to be installed."""

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.anthropic_model)
