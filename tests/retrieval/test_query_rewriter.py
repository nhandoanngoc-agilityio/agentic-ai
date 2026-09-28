from market_research_team.retrieval import query_rewriter as query_rewriter_module
from market_research_team.retrieval.query_rewriter import rewrite_and_expand


class _FakeQueryExpansion:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries


class _FakeStructuredLLM:
    def __init__(self, result: _FakeQueryExpansion) -> None:
        self._result = result

    def invoke(self, _messages: list[object], config: object = None) -> _FakeQueryExpansion:
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


def test_rewrite_and_expand_falls_back_to_objective_on_llm_error(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        query_rewriter_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )

    queries = rewrite_and_expand("Assess competitor pricing strategy", _BrokenLLM())  # type: ignore[arg-type]

    assert queries == ["Assess competitor pricing strategy"]
    assert len(calls) == 1
    assert calls[0]["component"] == "query_rewriter"
    assert calls[0]["reason"].startswith("exception:")


def test_rewrite_and_expand_falls_back_when_llm_returns_no_usable_queries(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        query_rewriter_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )
    llm = _FakeLLM(_FakeQueryExpansion(["   ", ""]))

    queries = rewrite_and_expand("Assess competitor pricing strategy", llm)  # type: ignore[arg-type]

    assert queries == ["Assess competitor pricing strategy"]
    assert len(calls) == 1
    assert calls[0]["reason"] == "empty_result"


def test_rewrite_and_expand_wraps_objective_in_delimiter_tags() -> None:
    captured: dict[str, list[object]] = {}

    class _CapturingStructuredLLM:
        def invoke(self, messages: list[object], config: object = None) -> _FakeQueryExpansion:
            captured["messages"] = messages
            return _FakeQueryExpansion(["acme pricing"])

    class _CapturingLLM:
        def with_structured_output(self, _schema: object) -> _CapturingStructuredLLM:
            return _CapturingStructuredLLM()

    rewrite_and_expand("Assess Acme pricing", _CapturingLLM())  # type: ignore[arg-type]

    human_message = captured["messages"][1]
    assert human_message.content == (
        "<research_objective>\nAssess Acme pricing\n</research_objective>"
    )
