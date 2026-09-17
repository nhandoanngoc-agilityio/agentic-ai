from langchain_core.messages import AIMessage

from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import mean, percent_change


class _ScriptedBoundLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages: list[object]) -> AIMessage:
        return self._responses.pop(0)


class _ScriptedLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses

    def bind_tools(self, _tools: list[object]) -> _ScriptedBoundLLM:
        return _ScriptedBoundLLM(self._responses)


class _RepeatingBoundLLM:
    def __init__(self, message: AIMessage) -> None:
        self._message = message
        self.call_count = 0

    def invoke(self, _messages: list[object]) -> AIMessage:
        self.call_count += 1
        return self._message


class _RepeatingLLM:
    def __init__(self, message: AIMessage) -> None:
        self._message = message
        self.bound: _RepeatingBoundLLM | None = None

    def bind_tools(self, _tools: list[object]) -> _RepeatingBoundLLM:
        self.bound = _RepeatingBoundLLM(self._message)
        return self.bound


def test_run_tool_calling_loop_records_successful_tool_calls_as_results() -> None:
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "mean", "args": {"values": [2, 4, 6]}, "id": "call-1", "type": "tool_call"}
        ],
    )
    final_message = AIMessage(content="Done.", tool_calls=[])
    llm = _ScriptedLLM([tool_call_message, final_message])

    results = run_tool_calling_loop(llm, [mean], "Compare pricing", [])  # type: ignore[arg-type]

    assert len(results) == 1
    assert results[0]["metric"] == "mean"
    assert results[0]["value"] == 4.0


def test_run_tool_calling_loop_reports_tool_errors_without_crashing() -> None:
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "percent_change",
                "args": {"old_value": 0, "new_value": 10},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final_message = AIMessage(content="Done.", tool_calls=[])
    llm = _ScriptedLLM([tool_call_message, final_message])

    results = run_tool_calling_loop(llm, [percent_change], "Compare pricing", [])  # type: ignore[arg-type]

    assert results == []


def test_run_tool_calling_loop_stops_at_max_iterations() -> None:
    always_tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "mean", "args": {"values": [1, 2, 3]}, "id": "call", "type": "tool_call"}
        ],
    )
    llm = _RepeatingLLM(always_tool_call)

    results = run_tool_calling_loop(
        llm,  # type: ignore[arg-type]
        [mean],
        "Compare pricing",
        [],
        max_iterations=2,
    )

    assert len(results) == 2
    assert llm.bound is not None
    assert llm.bound.call_count == 2
