from langchain_core.messages import AIMessage

from market_research_team.agents.supervisor import router as router_module
from market_research_team.agents.supervisor.router import decide_next_step
from market_research_team.state import AgentState

_FINDING = {"source": "x", "content": "y", "relevance_score": 1.0}
_RESULT = {"metric": "m", "value": 1.0, "detail": "d"}


class _FakeDecision:
    def __init__(self, next_step: str) -> None:
        self.next = next_step


class _FakeStructuredLLM:
    def __init__(self, result: _FakeDecision) -> None:
        self._result = result

    def invoke(self, _messages: list[object], config: object = None) -> _FakeDecision:
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
    from_response_cache: bool = False,
) -> AgentState:
    state: AgentState = {
        "messages": [
            AIMessage(content="Routing.", name="supervisor") for _ in range(supervisor_visits)
        ],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": research_findings or [],
        "analytics_results": analytics_results or [],
        "report_path": report_path,
    }
    if from_response_cache:
        state["from_response_cache"] = True
    return state


def test_error_in_state_short_circuits_to_finish_without_calling_llm() -> None:
    state = _state(research_findings=[_FINDING])
    state["error"] = "research failed: boom"

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "FINISH"


def test_routes_to_research_when_no_findings_without_calling_llm() -> None:
    result = decide_next_step(_state(), _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "research"


def test_response_cache_hit_routes_straight_to_reporting_without_calling_llm() -> None:
    state = _state(
        research_findings=[_FINDING], analytics_results=[_RESULT], from_response_cache=True
    )

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "reporting"


def test_response_cache_flag_without_analytics_results_falls_through_to_llm() -> None:
    state = _state(research_findings=[_FINDING], from_response_cache=True)

    result = decide_next_step(state, _FakeLLM("analytics"))  # type: ignore[arg-type]

    assert result == "analytics"


def test_response_cache_flag_is_ignored_once_report_already_written() -> None:
    state = _state(
        research_findings=[_FINDING],
        analytics_results=[_RESULT],
        report_path="reports/mock-report.md",
        from_response_cache=True,
    )

    result = decide_next_step(state, _ExplodingLLM())  # type: ignore[arg-type]

    assert result == "FINISH"


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


def test_clamps_invalid_llm_choice_to_the_safe_default(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        router_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )
    state = _state(research_findings=[_FINDING])

    result = decide_next_step(state, _FakeLLM("reporting"))  # type: ignore[arg-type]

    assert result == "research"
    assert len(calls) == 1
    assert calls[0]["component"] == "supervisor_router"
    assert calls[0]["reason"].startswith("invalid_llm_choice:")


def test_falls_back_to_safe_default_on_llm_error(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        router_module.audit,
        "record",
        lambda event, thread_id, **fields: calls.append({"event": event, **fields}),
    )
    state = _state(research_findings=[_FINDING])

    result = decide_next_step(state, _BrokenLLM())  # type: ignore[arg-type]

    assert result == "analytics"
    assert len(calls) == 1
    assert calls[0]["reason"].startswith("exception:")


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


class _FocusedDecision:
    def __init__(self, next_step: str, focus: str | None) -> None:
        self.next = next_step
        self.research_focus = focus


class _CapturingFocusLLM:
    """Returns a fixed decision with a focus and records the prompt it saw."""

    def __init__(self, next_step: str, focus: str | None) -> None:
        self._decision = _FocusedDecision(next_step, focus)
        self.messages: list[object] = []

    def with_structured_output(self, _schema: object) -> "_CapturingFocusLLM":
        return self

    def invoke(self, messages: list[object], config: object = None) -> _FocusedDecision:
        self.messages = messages
        return self._decision


def test_hand_back_to_research_carries_the_named_gap() -> None:
    from market_research_team.agents.supervisor.router import decide_route

    llm = _CapturingFocusLLM("research", "  Globex churn  ")
    route = decide_route(_state(research_findings=[_FINDING]), llm)  # type: ignore[arg-type]

    assert route.next == "research"
    assert route.research_focus == "Globex churn"
    # The supervisor sees what was found, not only a count, so it can name a gap.
    assert "<findings_summary>\n- [x] y\n</findings_summary>" in llm.messages[1].content


def test_focus_is_dropped_when_not_routing_to_research() -> None:
    from market_research_team.agents.supervisor.router import decide_route

    llm = _CapturingFocusLLM("analytics", "stray focus")
    route = decide_route(_state(research_findings=[_FINDING]), llm)  # type: ignore[arg-type]

    assert route == ("analytics", None)


def test_exhausted_research_is_not_offered_again() -> None:
    state = _state(research_findings=[_FINDING], analytics_results=[_RESULT])
    state["research_exhausted"] = True

    # Only analytics/reporting remain; an LLM that still says research is clamped.
    assert decide_next_step(state, _FakeLLM("research")) == "analytics"  # type: ignore[arg-type]


def test_exhausted_research_with_one_option_left_skips_the_llm() -> None:
    state = _state(research_findings=[_FINDING])
    state["research_exhausted"] = True

    assert decide_next_step(state, _ExplodingLLM()) == "analytics"  # type: ignore[arg-type]
