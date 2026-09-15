"""Unit tests for the Reporting Agent's report drafting and small helpers."""

from langchain_core.messages import AIMessage

from market_research_team.agents.reporting.node import _extract_tool_text, _slugify, draft_report

_FINDING = {
    "source": "competitor_acme.md",
    "content": "Acme prices at $49/seat.",
    "relevance_score": 0.9,
}
_RESULT = {"metric": "mean", "value": 49.0, "detail": "mean([49]) = 49.0"}


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> AIMessage:
        self.last_messages = messages
        return AIMessage(content=self._content)


class _BrokenLLM:
    def invoke(self, _messages: list[object]) -> AIMessage:
        raise RuntimeError("boom")


def test_draft_report_uses_llm_output_when_available() -> None:
    result = draft_report(
        "Assess pricing", [_FINDING], [_RESULT], _FakeLLM("# Custom Report")  # type: ignore[arg-type]
    )

    assert result == "# Custom Report"


def test_draft_report_falls_back_on_llm_error() -> None:
    result = draft_report("Assess pricing", [_FINDING], [_RESULT], _BrokenLLM())  # type: ignore[arg-type]

    assert "# Research Report" in result
    assert "Assess pricing" in result
    assert "Acme prices at $49/seat." in result
    assert "mean: 49.0" in result


def test_draft_report_falls_back_on_blank_llm_output() -> None:
    result = draft_report("Assess pricing", [_FINDING], [_RESULT], _FakeLLM("   "))  # type: ignore[arg-type]

    assert "# Research Report" in result


def test_draft_report_folds_reviewer_feedback_into_the_prompt() -> None:
    llm = _FakeLLM("# Revised Report")

    result = draft_report(
        "Assess pricing", [_FINDING], [_RESULT], llm, feedback="Add a pricing comparison table."  # type: ignore[arg-type]
    )

    assert result == "# Revised Report"
    assert llm.last_messages is not None
    human_message = llm.last_messages[1]
    assert "Add a pricing comparison table." in human_message.content


def test_draft_report_omits_feedback_section_when_none_given() -> None:
    llm = _FakeLLM("# Draft")

    draft_report("Assess pricing", [_FINDING], [_RESULT], llm)  # type: ignore[arg-type]

    assert llm.last_messages is not None
    human_message = llm.last_messages[1]
    assert "reviewer rejected" not in human_message.content


def test_draft_report_handles_no_findings_or_results() -> None:
    result = draft_report("Assess pricing", [], [], _BrokenLLM())  # type: ignore[arg-type]

    assert "No research findings were available." in result
    assert "No analytics were computed." in result


def test_slugify_produces_a_clean_filename_stem() -> None:
    assert (
        _slugify("Assess Acme vs Globex pricing strategy!!")
        == "assess-acme-vs-globex-pricing-strategy"
    )


def test_slugify_handles_empty_string() -> None:
    assert _slugify("") == "report"


def test_extract_tool_text_from_plain_string() -> None:
    assert _extract_tool_text("already a string") == "already a string"


def test_extract_tool_text_from_content_blocks() -> None:
    blocks = [{"type": "text", "text": "/tmp/report.md", "id": "abc"}]
    assert _extract_tool_text(blocks) == "/tmp/report.md"


def test_extract_tool_text_joins_multiple_blocks() -> None:
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _extract_tool_text(blocks) == "ab"
