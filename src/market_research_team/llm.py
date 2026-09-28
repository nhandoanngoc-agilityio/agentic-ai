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


def get_chat_model() -> BaseChatModel:
    """Construct the configured chat model. Provider packages are imported
    lazily so only the one actually selected needs to be installed.

    Binds `TokenUsageCallbackHandler` so every call site (query rewriting,
    supervisor routing, analytics tool-calling, report drafting) gets local
    token-usage tracking automatically, without each one needing to pass a
    callback itself.
    """

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        model: BaseChatModel = ChatOpenAI(model=settings.openai_model)
    else:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model=settings.anthropic_model)

    return model.with_config({"callbacks": [TokenUsageCallbackHandler()]})
