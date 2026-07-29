from langchain_core.messages import AIMessage

from market_research_team.state import AgentState
from market_research_team.supervisor import router as router_module
from market_research_team.supervisor.router import decide_next_step

_FINDING = {"source": "x", "content": "y", "relevance_score": 1.0}
_RESULT = {"metric": "m", "value": 1.0, "detail": "d"}


class _FakeDecision:
    def __init__(self, next_step: str) -> None:
        self.next = next_step


class _FakeStructuredLLM:
    def __init__(self, result: _FakeDecision) -> None:
        self._result = result

    def invoke(self, _messages: list[object]) -> _FakeDecision:
        return self._result


class _FakeLLM:
    def __init__(self, next_step: str) -> None:
        self._next_step = next_step

    def with_structured_output(self, _schema: object) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(_FakeDecision(self._next_step))


class _BrokenLLM:
    def with_structured_output(self, _schema: object) -> None:
        raise RuntimeError("boom")


class _ExplodingLLM:
    """An LLM that must never be invoked — proves a decision short-circuits."""

    def with_structured_output(self, _schema: object) -> None:
        raise AssertionError("the LLM should not have been consulted for this decision")


def _state(
    *,
    research_findings: list | None = None,
    analytics_results: list | None = None,
    report_path: str | None = None,
    supervisor_visits: int = 0,
) -> AgentState:
    return {
        "messages": [
            AIMessage(content="Routing.", name="supervisor") for _ in range(supervisor_visits)
        ],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": research_findings or [],
        "analytics_results": analytics_results or [],
        "report_path": report_path,
    }


def test_routes_to_research_when_no_findings_without_calling_llm() -> None:
    result = decide_next_step(_state(), _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "research"


def test_finishes_when_report_already_written_without_calling_llm() -> None:
    state = _state(
        research_findings=[_FINDING],
        analytics_results=[_RESULT],
        report_path="reports/mock-report.md",
    )

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "FINISH"


def test_asks_llm_to_choose_between_research_and_analytics() -> None:
    state = _state(research_findings=[_FINDING])

    assert decide_next_step(state, _FakeLLM("analytics")) == "analytics"  # type: ignore[arg-type]
    assert decide_next_step(state, _FakeLLM("research")) == "research"  # type: ignore[arg-type]


def test_asks_llm_to_choose_among_all_three_once_analytics_done() -> None:
    state = _state(research_findings=[_FINDING], analytics_results=[_RESULT])

    assert decide_next_step(state, _FakeLLM("reporting")) == "reporting"  # type: ignore[arg-type]
    assert decide_next_step(state, _FakeLLM("research")) == "research"  # type: ignore[arg-type]


def test_clamps_invalid_llm_choice_to_the_safe_default() -> None:
    state = _state(research_findings=[_FINDING])

    result = decide_next_step(state, _FakeLLM("reporting"))  # type: ignore[arg-type]

    assert result == "research"


def test_falls_back_to_safe_default_on_llm_error() -> None:
    state = _state(research_findings=[_FINDING])

    result = decide_next_step(state, _BrokenLLM())  # type: ignore[arg-type]

    assert result == "analytics"


def test_forces_progress_once_the_visit_cap_is_reached() -> None:
    state = _state(
        research_findings=[_FINDING],
        supervisor_visits=router_module._MAX_ROUTING_VISITS,
    )

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "analytics"


def test_forces_reporting_once_the_visit_cap_is_reached_and_analytics_is_done() -> None:
    state = _state(
        research_findings=[_FINDING],
        analytics_results=[_RESULT],
        supervisor_visits=router_module._MAX_ROUTING_VISITS,
    )

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "reporting"
