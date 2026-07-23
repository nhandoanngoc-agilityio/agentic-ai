from market_research_team.agents.research.query_rewriter import rewrite_and_expand


class _FakeQueryExpansion:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries


class _FakeStructuredLLM:
    def __init__(self, result: _FakeQueryExpansion) -> None:
        self._result = result

    def invoke(self, _messages: list[object]) -> _FakeQueryExpansion:
        return self._result


class _FakeLLM:
    def __init__(self, result: _FakeQueryExpansion) -> None:
        self._result = result

    def with_structured_output(self, _schema: object) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(self._result)


class _BrokenLLM:
    def with_structured_output(self, _schema: object) -> None:
        raise RuntimeError("boom")


def test_rewrite_and_expand_returns_cleaned_llm_queries() -> None:
    llm = _FakeLLM(_FakeQueryExpansion(["acme pricing", "acme security", "  "]))

    queries = rewrite_and_expand("How does Acme compare on pricing and security?", llm)  # type: ignore[arg-type]

    assert queries == ["acme pricing", "acme security"]


def test_rewrite_and_expand_falls_back_to_objective_on_llm_error() -> None:
    queries = rewrite_and_expand("Assess competitor pricing strategy", _BrokenLLM())  # type: ignore[arg-type]

    assert queries == ["Assess competitor pricing strategy"]


def test_rewrite_and_expand_falls_back_when_llm_returns_no_usable_queries() -> None:
    llm = _FakeLLM(_FakeQueryExpansion(["   ", ""]))

    queries = rewrite_and_expand("Assess competitor pricing strategy", llm)  # type: ignore[arg-type]

    assert queries == ["Assess competitor pricing strategy"]
