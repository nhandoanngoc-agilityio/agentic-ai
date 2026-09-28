from gradio_app.app import record_satisfaction_rating, resolve_interrupt, submit_objective
from langgraph.types import Command

from market_research_team.security import audit
from market_research_team.state import AgentState


class _FakeCompiledGraph:
    """Stands in for a compiled LangGraph graph: `invoke` returns a scripted
    result instead of running real nodes, keeping this test hermetic."""

    def __init__(self, result: AgentState) -> None:
        self._result = result

    def invoke(self, _state_or_command: object, config: object) -> AgentState:
        return self._result


def test_submit_objective_renders_finished_run_without_interrupt() -> None:
    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "FINISH",
        "research_findings": [],
        "analytics_results": [
            {"metric": "mean", "value": 10.0, "detail": "", "entity": "Acme"},
            {"metric": "mean", "value": 25.0, "detail": "", "entity": "Globex"},
        ],
        "report_path": "/tmp/report.md",
        "guardrail_events": [],
    }
    graph = _FakeCompiledGraph(fake_result)

    history, thread_id, pending_interrupt, figure, approval_visible, ran_objective, concluded = (
        submit_objective("Compare Acme vs Globex pricing", [], None, graph)
    )

    assert thread_id  # a thread id was generated
    assert pending_interrupt is None
    assert approval_visible is False
    assert figure is not None
    assert ran_objective == "Compare Acme vs Globex pricing"
    assert concluded is True
    assert any("report.md" in str(turn.get("content", "")) for turn in history)


def test_submit_objective_surfaces_pending_interrupt() -> None:
    class _Interrupt:
        def __init__(self, value: dict) -> None:
            self.value = value

    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "reporting",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
        "guardrail_events": [],
        "__interrupt__": [
            _Interrupt(
                {
                    "action": "write_report",
                    "filename": "x.md",
                    "content": "draft",
                    "attempt": 1,
                    "max_attempts": 3,
                    "warnings": [],
                }
            )
        ],
    }
    graph = _FakeCompiledGraph(fake_result)

    history, thread_id, pending_interrupt, figure, approval_visible, ran_objective, concluded = (
        submit_objective("Compare Acme vs Globex pricing", [], None, graph)
    )

    assert pending_interrupt == {
        "action": "write_report",
        "filename": "x.md",
        "content": "draft",
        "attempt": 1,
        "max_attempts": 3,
        "warnings": [],
    }
    assert approval_visible is True
    assert concluded is False
    assert any("draft" in str(turn.get("content", "")) for turn in history)


def test_submit_objective_surfaces_guardrail_warnings_and_round_info() -> None:
    class _Interrupt:
        def __init__(self, value: dict) -> None:
            self.value = value

    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "reporting",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
        "guardrail_events": [],
        "__interrupt__": [
            _Interrupt(
                {
                    "action": "write_report",
                    "filename": "acme-vs-globex.md",
                    "content": "draft body",
                    "attempt": 2,
                    "max_attempts": 3,
                    "warnings": ["redacted an email address", "marked a figure [unverified]"],
                }
            )
        ],
    }
    graph = _FakeCompiledGraph(fake_result)

    history, _thread_id, _pending, _figure, _visible, _ran_objective, _concluded = submit_objective(
        "Compare Acme vs Globex pricing", [], None, graph
    )

    rendered = str(history[-1]["content"])
    assert "acme-vs-globex.md" in rendered
    assert "2/3" in rendered
    assert "redacted an email address" in rendered
    assert "marked a figure [unverified]" in rendered
    assert "draft body" in rendered


def test_submit_objective_mints_a_fresh_thread_id_each_call() -> None:
    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "FINISH",
        "research_findings": [],
        "analytics_results": [],
        "report_path": "/tmp/report.md",
        "guardrail_events": [],
    }
    graph = _FakeCompiledGraph(fake_result)

    _history1, thread_id_1, *_ = submit_objective("Compare Acme vs Globex pricing", [], None, graph)
    # Simulate the UI passing back a "stale" thread_id from a prior run, the
    # way `gr.State` would if it weren't reset between fresh objectives.
    _history2, thread_id_2, *_ = submit_objective(
        "Compare Acme vs Globex pricing", [], thread_id_1, graph
    )

    assert thread_id_1 != thread_id_2


class _RaisingGraph:
    """Stands in for a compiled graph whose `invoke` blows up, so tests can
    verify the caller renders an error turn instead of propagating."""

    def invoke(self, _state_or_command: object, config: object) -> AgentState:
        raise RuntimeError("boom")


def test_submit_objective_renders_invalid_objective_as_error_turn() -> None:
    graph = _FakeCompiledGraph({})  # never invoked; validation fails first

    history, thread_id, pending_interrupt, figure, approval_visible, ran_objective, concluded = (
        submit_objective("", [], None, graph)
    )

    assert pending_interrupt is None
    assert figure is None
    assert approval_visible is False
    assert concluded is True
    assert any("Run failed" in str(turn.get("content", "")) for turn in history)
    # The invalid input itself must still be visible above the error, so the
    # transcript explains what triggered the failure.
    assert any(turn.get("role") == "user" and turn.get("content") == "" for turn in history)


def test_submit_objective_renders_run_graph_exception_as_error_turn() -> None:
    graph = _RaisingGraph()

    history, thread_id, pending_interrupt, figure, approval_visible, ran_objective, concluded = (
        submit_objective("Compare Acme vs Globex pricing", [], None, graph)
    )

    assert pending_interrupt is None
    assert figure is None
    assert approval_visible is False
    assert concluded is True
    assert any("Run failed" in str(turn.get("content", "")) for turn in history)


class _ResumeCapturingGraph:
    """Records what it was invoked with, so tests can assert the resume
    decision was passed through as a `Command`, and returns a scripted
    finished-run result."""

    def __init__(self, result: AgentState) -> None:
        self._result = result
        self.last_invoke_arg: object = None

    def invoke(self, state_or_command: object, config: object) -> AgentState:
        self.last_invoke_arg = state_or_command
        return self._result


def test_resolve_interrupt_resumes_with_a_command() -> None:
    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "FINISH",
        "research_findings": [],
        "analytics_results": [],
        "report_path": "/tmp/report.md",
        "guardrail_events": [],
    }
    graph = _ResumeCapturingGraph(fake_result)

    history, pending_interrupt, figure, approval_visible, concluded = resolve_interrupt(
        {"approved": True}, [], "thread-1", graph
    )

    assert isinstance(graph.last_invoke_arg, Command)
    assert graph.last_invoke_arg.resume == {"approved": True}
    assert pending_interrupt is None
    assert approval_visible is False
    assert concluded is True
    assert any("report.md" in str(turn.get("content", "")) for turn in history)


def test_resolve_interrupt_surfaces_a_second_round_interrupt() -> None:
    class _Interrupt:
        def __init__(self, value: dict) -> None:
            self.value = value

    fake_result: AgentState = {
        "messages": [],
        "objective": "Compare Acme vs Globex pricing",
        "next": "reporting",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
        "guardrail_events": [],
        "__interrupt__": [
            _Interrupt(
                {
                    "action": "write_report",
                    "filename": "x.md",
                    "content": "revised draft",
                    "attempt": 2,
                    "max_attempts": 3,
                    "warnings": [],
                }
            )
        ],
    }
    graph = _ResumeCapturingGraph(fake_result)

    history, pending_interrupt, figure, approval_visible, concluded = resolve_interrupt(
        {"approved": False, "feedback": "add pricing detail"}, [], "thread-1", graph
    )

    assert pending_interrupt["attempt"] == 2
    assert approval_visible is True
    assert concluded is False


def test_resolve_interrupt_renders_run_graph_exception_as_error_turn() -> None:
    graph = _RaisingGraph()

    history, pending_interrupt, figure, approval_visible, concluded = resolve_interrupt(
        {"approved": True}, [], "thread-1", graph
    )

    assert pending_interrupt is None
    assert figure is None
    assert approval_visible is False
    assert concluded is True
    assert any("Run failed" in str(turn.get("content", "")) for turn in history)


def test_record_satisfaction_rating_writes_to_audit_log(tmp_path, monkeypatch) -> None:
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit.settings, "audit_log_path", log)

    record_satisfaction_rating("thread-1", "Compare Acme vs Globex pricing", "up")

    rate, count = audit.satisfaction_rate(log)
    assert count == 1
    assert rate == 1.0
