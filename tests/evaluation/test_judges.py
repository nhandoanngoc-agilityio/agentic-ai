"""Tests for the shared LLM-judge functions (fake structured-output LLMs only)."""

from market_research_team.evaluation import judges


class _FakeStructured:
    def __init__(self, score: float) -> None:
        self.score = score
        self.messages: list[object] = []

    def invoke(self, messages: list[object]) -> judges.JudgeScoreSchema:
        self.messages = messages
        return judges.JudgeScoreSchema(passed=True, score=self.score, reasoning="ok")


class _FakeJudgeLLM:
    def __init__(self, score: float) -> None:
        self.structured = _FakeStructured(score)

    def with_structured_output(self, schema: object) -> _FakeStructured:
        assert schema is judges.JudgeScoreSchema
        return self.structured


def test_each_judge_uses_its_own_prompt_and_returns_the_score():
    cases = [
        (judges.judge_query_rewrite, ("obj", ["q1"]), judges.QUERY_REWRITE_JUDGE_PROMPT),
        (judges.judge_supervisor_decision, (1, 0, "analytics"), judges.SUPERVISOR_JUDGE_PROMPT),
        (judges.judge_analytics, ([], []), judges.ANALYTICS_JUDGE_PROMPT),
        (judges.judge_report, ("obj", [], [], "# R"), judges.REPORTING_JUDGE_PROMPT),
        (judges.judge_full_pipeline, ("obj", "# R"), judges.FULL_PIPELINE_JUDGE_PROMPT),
    ]
    for judge, args, prompt in cases:
        llm = _FakeJudgeLLM(0.8)
        result = judge(*args, llm)
        assert result.score == 0.8
        assert llm.structured.messages[0].content == prompt


def test_judge_prompts_hash_is_stable_and_short():
    assert judges.judge_prompts_hash() == judges.judge_prompts_hash()
    assert len(judges.judge_prompts_hash()) == 12


def test_judge_id_changes_with_source_and_model():
    local = judges.judge_id("local", "anthropic", "claude-opus-5")

    assert judges.judge_prompts_hash() in local
    assert local != judges.judge_id("langsmith", "anthropic", "claude-opus-5")
    assert local != judges.judge_id("local", "anthropic", "claude-sonnet-5")
