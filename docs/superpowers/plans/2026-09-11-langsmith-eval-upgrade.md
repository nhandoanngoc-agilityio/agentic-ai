# LangSmith Eval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangSmith LLM-judge scoring and dataset/experiment tracking on top of the existing
deterministic golden-dataset eval suite, behind an opt-in `--langsmith` flag on
`scripts/run_evals.py`, and restore CI.

**Architecture:** A new `evaluation/langsmith_eval.py` module mirrors each of the 5 existing eval
categories into a LangSmith Dataset, then runs `langsmith.evaluate()` per category with two
evaluators each: the existing deterministic `checks.py` logic (reused, not duplicated) and a new
category-tailored LLM-judge using the project's existing `get_chat_model()` +
`with_structured_output()` pattern. `scripts/run_evals.py` gains a `--langsmith` flag that runs
this additively alongside the unchanged local JSON-report flow. `.gitlab-ci.yml` is restored to
its pre-`3d08014` content (hermetic `ruff` + `pytest` only).

**Tech Stack:** Python 3.11+, LangGraph/LangChain, `langsmith` SDK (`Client`, `evaluate`), pydantic
structured output, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-langsmith-eval-upgrade-design.md`

## Global Constraints

- No changes to the hermetic `pytest` suite's existing fakes/mocks or to `offline_eval.py`'s
  public behavior — `scripts/run_evals.py` (no flag) must behave identically to today.
- No new `Settings` fields for LangSmith credentials — read `LANGSMITH_API_KEY` /
  `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` from the environment, same as
  the `langsmith` SDK and LangChain's tracer already do.
- No change to `data/eval_results/` JSON output format.
- Follow existing code style exactly: private `_XxxSchema(BaseModel)` + `Field(description=...)`
  for structured output (see `agents/research/query_rewriter.py`, `supervisor/router.py`);
  `SystemMessage`/`HumanMessage` for prompts; every LLM-driven function takes `llm: BaseChatModel`
  rather than building one internally.
- No `from __future__ import annotations` — this codebase writes native `list[str]` / `X | None`
  syntax everywhere (Python 3.11+ target), never `__future__` imports.
- `ruff check src tests scripts` must stay clean throughout (rules: `E`, `F`, `I`, `UP`,
  line-length 100).

---

### Task 1: LangSmith dependency, data model, and dataset sync

**Files:**
- Modify: `pyproject.toml` (add `langsmith` dependency)
- Create: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Produces (used by Tasks 2-7): `JudgeScore` dataclass, `_JudgeScoreSchema` pydantic model
  (structured-output schema — same shape as `JudgeScore`: `passed: bool`, `score: float`,
  `reasoning: str`), `sync_dataset(client, category: str, examples: list[dict[str, Any]]) ->
  Dataset`.
- Produces (used by Task 8): `LangSmithEvalSummary` dataclass (`category: str`,
  `experiment_name: str`, `pass_rate: float`).

- [ ] **Step 1: Add the `langsmith` dependency**

In `pyproject.toml`, in the `[project] dependencies` list, add a new line under the `# --- MCP
integration ---` block's neighboring section — add it to the `# --- LangChain core + model
provider ---` block since it's a first-party LangChain-ecosystem client:

```toml
    # --- LangChain core + model provider ---
    "langchain>=1.3.14,<2.0",
    "langchain-core>=0.4",
    "langchain-anthropic>=0.4",           # Claude provider
    "langchain-openai>=1.4.1",             # OpenAI provider (LLM_PROVIDER=openai)
    "langchain-text-splitters>=0.4",      # recursive/hierarchical chunking
    "langsmith>=0.6,<1.0",                # LangSmith client: tracing + eval datasets/experiments
```

- [ ] **Step 2: Reinstall to pull in the new dependency**

Run: `pip install -e ".[dev]"`
Expected: installs/upgrades `langsmith` with no errors.

- [ ] **Step 3: Write failing tests for `sync_dataset`**

Create `tests/test_langsmith_eval.py`:

```python
"""Hermetic tests for the LangSmith eval integration -- no network, no real
LangSmith calls. A fake Client double stands in for langsmith.Client so
sync_dataset's get-or-create-and-wipe logic is exercised deterministically.
"""

from market_research_team.evaluation import langsmith_eval


class _FakeDataset:
    def __init__(self, name: str) -> None:
        self.id = f"dataset-{name}"
        self.name = name


class _FakeExampleRecord:
    def __init__(self, example_id: str) -> None:
        self.id = example_id


class _FakeLangSmithClient:
    """Hermetic stand-in for langsmith.Client -- records calls instead of hitting the network."""

    def __init__(
        self, *, existing_dataset: bool = False, existing_example_ids: list[str] | None = None
    ) -> None:
        self._existing_dataset = existing_dataset
        self._example_ids = list(existing_example_ids or [])
        self.deleted_example_ids: list[str] = []
        self.created_examples: list[dict] = []
        self.created_dataset_names: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self._existing_dataset

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> _FakeDataset:
        self.created_dataset_names.append(dataset_name)
        return _FakeDataset(dataset_name)

    def read_dataset(self, *, dataset_name: str) -> _FakeDataset:
        return _FakeDataset(dataset_name)

    def list_examples(self, *, dataset_name: str) -> list[_FakeExampleRecord]:
        return [_FakeExampleRecord(example_id) for example_id in self._example_ids]

    def delete_example(self, *, example_id: str) -> None:
        self.deleted_example_ids.append(example_id)

    def create_example(self, *, inputs: dict, outputs: dict, dataset_id: str) -> None:
        self.created_examples.append({"inputs": inputs, "outputs": outputs, "dataset_id": dataset_id})


def test_sync_dataset_creates_new_dataset_when_absent() -> None:
    client = _FakeLangSmithClient(existing_dataset=False)
    examples = [{"inputs": {"objective": "x"}, "outputs": {"min_queries": 1}}]

    dataset = langsmith_eval.sync_dataset(client, "query_rewrite", examples)  # type: ignore[arg-type]

    assert client.created_dataset_names == ["market-research-team-query_rewrite"]
    assert client.created_examples == [
        {"inputs": {"objective": "x"}, "outputs": {"min_queries": 1}, "dataset_id": dataset.id}
    ]
    assert client.deleted_example_ids == []


def test_sync_dataset_wipes_existing_examples_before_recreating() -> None:
    client = _FakeLangSmithClient(existing_dataset=True, existing_example_ids=["ex-1", "ex-2"])
    examples = [{"inputs": {}, "outputs": {}}]

    langsmith_eval.sync_dataset(client, "analytics", examples)  # type: ignore[arg-type]

    assert client.deleted_example_ids == ["ex-1", "ex-2"]
    assert client.created_dataset_names == []
    assert len(client.created_examples) == 1
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError: module ... has no attribute
'sync_dataset'` (the module doesn't exist yet).

- [ ] **Step 5: Implement the module**

Create `src/market_research_team/evaluation/langsmith_eval.py`:

```python
"""LangSmith integration for the prompt evaluation regression suite: dataset sync and
category-tailored LLM-judge evaluators, layered on top of -- not replacing -- the deterministic
checks in evaluation/checks.py. Powers the --langsmith flag on scripts/run_evals.py.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class JudgeScore:
    passed: bool
    score: float
    reasoning: str


@dataclass
class LangSmithEvalSummary:
    category: str
    # From ExperimentResults.experiment_name() -- the SDK doesn't hand back a ready-made UI
    # URL, so callers report the name (findable in the LangSmith UI) instead.
    experiment_name: str
    pass_rate: float


class _JudgeScoreSchema(BaseModel):
    """Structured-output schema for every llm_judge_* evaluator -- same shape as JudgeScore."""

    passed: bool = Field(description="Whether the output meets the judge's quality bar.")
    score: float = Field(description="A 0.0-1.0 quality score, 1.0 being the best.")
    reasoning: str = Field(description="Brief justification for the score.")


def sync_dataset(client: Any, category: str, examples: list[dict[str, Any]]) -> Any:
    """Get-or-create `market-research-team-{category}`, wipe its examples, recreate from the
    current in-repo cases. LangSmith is always a mirror of golden_dataset.py, never a second
    source of truth -- this makes every --langsmith run reflect the latest local cases with no
    drift or manual dataset editing in the LangSmith UI."""

    dataset_name = f"market-research-team-{category}"
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        for existing in client.list_examples(dataset_name=dataset_name):
            client.delete_example(example_id=existing.id)
    else:
        dataset = client.create_dataset(
            dataset_name, description=f"Golden dataset mirror: {category}"
        )

    for example in examples:
        client.create_example(
            inputs=example["inputs"], outputs=example["outputs"], dataset_id=dataset.id
        )
    return dataset
```

Note: `client: Any` / return `Any` deliberately, not `langsmith.Client` / `langsmith.schemas.Dataset`
— the whole point of this function is to work against the fake test double above and the real
SDK client identically (structural typing), and the real types get imported into this module in
Step 6+ of Task 7 anyway once `run_langsmith_eval` needs them.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean, no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add langsmith dependency and dataset-sync foundation"
```

---

### Task 2: `query_rewrite` category (target, deterministic evaluator, LLM judge)

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `_JudgeScoreSchema` (Task 1), `checks.check_query_count` /
  `checks.check_keyword_coverage` (existing, `evaluation/checks.py`), `rewrite_and_expand(objective:
  str, llm: BaseChatModel) -> list[str]` (existing, `agents/research/query_rewriter.py`),
  `QUERY_REWRITE_CASES` (existing, `evaluation/golden_dataset.py`).
- Produces (used by Task 7): `_query_rewrite_examples() -> list[dict[str, Any]]`,
  `target_query_rewrite(inputs, *, llm) -> dict[str, Any]`,
  `deterministic_evaluator_query_rewrite(run, example) -> dict[str, Any]`,
  `llm_judge_query_rewrite(run, example, *, llm) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_langsmith_eval.py`:

```python
class _FakeRun:
    def __init__(self, outputs: dict | None) -> None:
        self.outputs = outputs


class _FakeExample:
    def __init__(self, inputs: dict | None = None, outputs: dict | None = None) -> None:
        self.inputs = inputs
        self.outputs = outputs


class _FakeStructuredJudgeLLM:
    """Satisfies `llm.with_structured_output(_JudgeScoreSchema).invoke(...)`."""

    def __init__(self, score: float, passed: bool, reasoning: str = "because") -> None:
        self._result = langsmith_eval._JudgeScoreSchema(passed=passed, score=score, reasoning=reasoning)

    def with_structured_output(self, _schema: object) -> "_FakeStructuredJudgeLLM":
        return self

    def invoke(self, _messages: list[object]) -> langsmith_eval._JudgeScoreSchema:
        return self._result


def test_query_rewrite_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import QUERY_REWRITE_CASES

    examples = langsmith_eval._query_rewrite_examples()

    assert len(examples) == len(QUERY_REWRITE_CASES)
    assert examples[0]["inputs"] == {"objective": QUERY_REWRITE_CASES[0].objective}
    assert examples[0]["outputs"]["min_queries"] == QUERY_REWRITE_CASES[0].min_queries


def test_target_query_rewrite_calls_rewrite_and_expand(monkeypatch) -> None:
    monkeypatch.setattr(
        langsmith_eval, "rewrite_and_expand", lambda objective, llm: [f"{objective}-q1"]
    )

    result = langsmith_eval.target_query_rewrite({"objective": "acme pricing"}, llm=object())

    assert result == {"queries": ["acme pricing-q1"]}


def test_deterministic_evaluator_query_rewrite_passes_on_good_output() -> None:
    run = _FakeRun(outputs={"queries": ["acme pricing", "globex pricing"]})
    example = _FakeExample(outputs={"min_queries": 1, "max_queries": 4, "required_any_keywords": ["pricing"]})

    result = langsmith_eval.deterministic_evaluator_query_rewrite(run, example)  # type: ignore[arg-type]

    assert result["key"] == "deterministic"
    assert result["score"] == 1.0


def test_deterministic_evaluator_query_rewrite_fails_without_keyword_coverage() -> None:
    run = _FakeRun(outputs={"queries": ["unrelated topic"]})
    example = _FakeExample(outputs={"min_queries": 1, "max_queries": 4, "required_any_keywords": ["pricing"]})

    result = langsmith_eval.deterministic_evaluator_query_rewrite(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_query_rewrite_returns_judge_score() -> None:
    run = _FakeRun(outputs={"queries": ["acme pricing"]})
    example = _FakeExample(inputs={"objective": "Compare Acme and Globex pricing"})
    fake_llm = _FakeStructuredJudgeLLM(score=0.9, passed=True, reasoning="good coverage")

    result = langsmith_eval.llm_judge_query_rewrite(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.9, "comment": "good coverage"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k query_rewrite`
Expected: FAIL with `AttributeError` (functions don't exist yet).

- [ ] **Step 3: Implement**

Append to `src/market_research_team/evaluation/langsmith_eval.py` (add the new imports to the
existing top-of-file import block):

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from market_research_team.agents.research.query_rewriter import rewrite_and_expand
from market_research_team.evaluation import checks
from market_research_team.evaluation.golden_dataset import QUERY_REWRITE_CASES
```

Then append the category implementation:

```python
def _query_rewrite_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"objective": case.objective},
            "outputs": {
                "min_queries": case.min_queries,
                "max_queries": case.max_queries,
                "required_any_keywords": case.required_any_keywords,
            },
        }
        for case in QUERY_REWRITE_CASES
    ]


def target_query_rewrite(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    queries = rewrite_and_expand(inputs["objective"], llm)
    return {"queries": queries}


def deterministic_evaluator_query_rewrite(run: Any, example: Any) -> dict[str, Any]:
    queries = (run.outputs or {}).get("queries", [])
    expected = example.outputs or {}
    count_passed, count_detail = checks.check_query_count(
        queries,
        min_count=expected.get("min_queries", 1),
        max_count=expected.get("max_queries", 6),
    )
    keyword_passed, keyword_detail = checks.check_keyword_coverage(
        queries, expected.get("required_any_keywords", [])
    )
    passed = count_passed and keyword_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{count_detail}; {keyword_detail}",
    }


_QUERY_REWRITE_JUDGE_PROMPT = (
    "You are grading whether a set of search sub-queries are good, non-redundant "
    "decompositions of a research objective for a market and competitor research "
    "assistant. Score 1.0 if the queries clearly cover distinct angles of the "
    "objective, lower if they're redundant, off-topic, or too vague."
)


def llm_judge_query_rewrite(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    queries = (run.outputs or {}).get("queries", [])
    objective = (example.inputs or {}).get("objective", "")
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_QUERY_REWRITE_JUDGE_PROMPT),
            HumanMessage(content=f"Objective: {objective}\nQueries: {queries}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add query_rewrite LangSmith target/evaluators"
```

---

### Task 3: `supervisor_decision` category

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `decide_next_step(state: AgentState, llm: BaseChatModel) -> RouteDecision` (existing,
  `supervisor/router.py`), `SUPERVISOR_DECISION_CASES` (existing, `evaluation/golden_dataset.py`),
  `AgentState` (existing, `state.py`), `_JudgeScoreSchema` / `_FakeRun` / `_FakeExample` /
  `_FakeStructuredJudgeLLM` (Task 1 & 2).
- Produces (used by Task 7): `_supervisor_decision_examples() -> list[dict[str, Any]]`,
  `target_supervisor_decision(inputs, *, llm) -> dict[str, Any]`,
  `deterministic_evaluator_supervisor_decision(run, example) -> dict[str, Any]`,
  `llm_judge_supervisor_decision(run, example, *, llm) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_langsmith_eval.py`:

```python
def test_supervisor_decision_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import SUPERVISOR_DECISION_CASES

    examples = langsmith_eval._supervisor_decision_examples()

    assert len(examples) == len(SUPERVISOR_DECISION_CASES)
    assert examples[0]["outputs"]["allowed_decisions"] == list(
        SUPERVISOR_DECISION_CASES[0].allowed_decisions
    )


def test_target_supervisor_decision_calls_decide_next_step(monkeypatch) -> None:
    monkeypatch.setattr(langsmith_eval, "decide_next_step", lambda state, llm: "analytics")

    result = langsmith_eval.target_supervisor_decision(
        {"research_findings": [], "analytics_results": [], "report_path": None}, llm=object()
    )

    assert result == {"decision": "analytics"}


def test_deterministic_evaluator_supervisor_decision_passes_on_allowed_choice() -> None:
    run = _FakeRun(outputs={"decision": "analytics"})
    example = _FakeExample(outputs={"allowed_decisions": ["research", "analytics"]})

    result = langsmith_eval.deterministic_evaluator_supervisor_decision(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_supervisor_decision_fails_on_disallowed_choice() -> None:
    run = _FakeRun(outputs={"decision": "reporting"})
    example = _FakeExample(outputs={"allowed_decisions": ["research", "analytics"]})

    result = langsmith_eval.deterministic_evaluator_supervisor_decision(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_supervisor_decision_returns_judge_score() -> None:
    run = _FakeRun(outputs={"decision": "analytics"})
    example = _FakeExample(inputs={"research_findings": [{"source": "x"}], "analytics_results": []})
    fake_llm = _FakeStructuredJudgeLLM(score=0.8, passed=True, reasoning="reasonable")

    result = langsmith_eval.llm_judge_supervisor_decision(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.8, "comment": "reasonable"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k supervisor_decision`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to the import block in `langsmith_eval.py`:

```python
from market_research_team.evaluation.golden_dataset import SUPERVISOR_DECISION_CASES
from market_research_team.state import AgentState
from market_research_team.supervisor.router import decide_next_step
```

Append:

```python
def _supervisor_decision_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "research_findings": case.research_findings,
                "analytics_results": case.analytics_results,
                "report_path": case.report_path,
            },
            "outputs": {"allowed_decisions": list(case.allowed_decisions)},
        }
        for case in SUPERVISOR_DECISION_CASES
    ]


def target_supervisor_decision(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    state: AgentState = {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": inputs["research_findings"],
        "analytics_results": inputs["analytics_results"],
        "report_path": inputs["report_path"],
    }
    decision = decide_next_step(state, llm)
    return {"decision": decision}


def deterministic_evaluator_supervisor_decision(run: Any, example: Any) -> dict[str, Any]:
    decision = (run.outputs or {}).get("decision")
    allowed = (example.outputs or {}).get("allowed_decisions", [])
    passed = decision in allowed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"decision={decision!r} (allowed: {allowed})",
    }


_SUPERVISOR_JUDGE_PROMPT = (
    "You are grading whether a supervisor's routing decision for a market research "
    "multi-agent system is reasonable given the current state, not just technically "
    "allowed. Score 1.0 if the decision clearly makes sense given what's been "
    "gathered so far."
)


def llm_judge_supervisor_decision(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    decision = (run.outputs or {}).get("decision")
    findings_count = len((example.inputs or {}).get("research_findings", []))
    results_count = len((example.inputs or {}).get("analytics_results", []))
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_SUPERVISOR_JUDGE_PROMPT),
            HumanMessage(
                content=(
                    f"Research findings so far: {findings_count}\n"
                    f"Analytics results so far: {results_count}\n"
                    f"Decision made: {decision!r}"
                )
            ),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add supervisor_decision LangSmith target/evaluators"
```

---

### Task 4: `analytics` category

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `run_tool_calling_loop(llm, tools, objective, findings, *, max_iterations=4) ->
  list[AnalyticsResult]` (existing, `agents/analytics/node.py`), `ANALYTICS_TOOLS` (existing,
  `agents/analytics/tools.py`), `checks.check_min_length` / `checks.check_any_value_matches`
  (existing), `ANALYTICS_CASES` (existing, `evaluation/golden_dataset.py`).
- Produces (used by Task 7): `_analytics_examples() -> list[dict[str, Any]]`,
  `target_analytics(inputs, *, llm) -> dict[str, Any]`,
  `deterministic_evaluator_analytics(run, example) -> dict[str, Any]`,
  `llm_judge_analytics(run, example, *, llm) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_langsmith_eval.py`:

```python
def test_analytics_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import ANALYTICS_CASES

    examples = langsmith_eval._analytics_examples()

    assert len(examples) == len(ANALYTICS_CASES)
    assert examples[0]["outputs"]["plausible_values"] == ANALYTICS_CASES[0].plausible_values


def test_target_analytics_calls_run_tool_calling_loop(monkeypatch) -> None:
    fake_results = [{"metric": "mean", "value": 275000.0, "detail": "d"}]
    monkeypatch.setattr(
        langsmith_eval,
        "run_tool_calling_loop",
        lambda llm, tools, objective, findings: fake_results,
    )

    result = langsmith_eval.target_analytics({"objective": "x", "findings": []}, llm=object())

    assert result == {"results": fake_results}


def test_deterministic_evaluator_analytics_passes_on_grounded_value() -> None:
    run = _FakeRun(outputs={"results": [{"metric": "mean", "value": 275000.0, "detail": "d"}]})
    example = _FakeExample(outputs={"min_tool_calls": 1, "plausible_values": [275000.0], "tolerance": 1.0})

    result = langsmith_eval.deterministic_evaluator_analytics(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_analytics_fails_when_no_results() -> None:
    run = _FakeRun(outputs={"results": []})
    example = _FakeExample(outputs={"min_tool_calls": 1, "plausible_values": [275000.0], "tolerance": 1.0})

    result = langsmith_eval.deterministic_evaluator_analytics(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_analytics_returns_judge_score() -> None:
    run = _FakeRun(outputs={"results": [{"metric": "mean", "value": 275000.0, "detail": "d"}]})
    example = _FakeExample(inputs={"findings": [{"source": "x", "content": "y"}]})
    fake_llm = _FakeStructuredJudgeLLM(score=1.0, passed=True, reasoning="grounded")

    result = langsmith_eval.llm_judge_analytics(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 1.0, "comment": "grounded"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k analytics`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to the import block:

```python
from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import ANALYTICS_TOOLS
from market_research_team.evaluation.golden_dataset import ANALYTICS_CASES
```

Append:

```python
def _analytics_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"objective": case.objective, "findings": case.findings},
            "outputs": {
                "min_tool_calls": case.min_tool_calls,
                "plausible_values": case.plausible_values,
                "tolerance": case.tolerance,
            },
        }
        for case in ANALYTICS_CASES
    ]


def target_analytics(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    results = run_tool_calling_loop(llm, ANALYTICS_TOOLS, inputs["objective"], inputs["findings"])
    return {"results": results}


def deterministic_evaluator_analytics(run: Any, example: Any) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    expected = example.outputs or {}
    count_passed, count_detail = checks.check_min_length(
        results, expected.get("min_tool_calls", 1), "tool calls"
    )
    value_passed, value_detail = checks.check_any_value_matches(
        results, expected.get("plausible_values", []), expected.get("tolerance", 1.0)
    )
    passed = count_passed and value_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{count_detail}; {value_detail}",
    }


_ANALYTICS_JUDGE_PROMPT = (
    "You are grading whether computed analytics results are grounded in the given "
    "research findings for a market research system. Score 1.0 only if every "
    "reported value plausibly traces back to a number stated in the findings -- "
    "score low if any value looks invented."
)


def llm_judge_analytics(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    results = (run.outputs or {}).get("results", [])
    findings = (example.inputs or {}).get("findings", [])
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_ANALYTICS_JUDGE_PROMPT),
            HumanMessage(content=f"Findings: {findings}\nComputed results: {results}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add analytics LangSmith target/evaluators"
```

---

### Task 5: `reporting` category

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `draft_report(objective, findings, results, llm) -> str` (existing,
  `agents/reporting/node.py`), `checks.check_contains_all` (existing), `REPORTING_CASES`
  (existing, `evaluation/golden_dataset.py`).
- Produces (used by Task 7): `_reporting_examples() -> list[dict[str, Any]]`,
  `target_reporting(inputs, *, llm) -> dict[str, Any]`,
  `deterministic_evaluator_reporting(run, example) -> dict[str, Any]`,
  `llm_judge_reporting(run, example, *, llm) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_langsmith_eval.py`:

```python
def test_reporting_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import REPORTING_CASES

    examples = langsmith_eval._reporting_examples()

    assert len(examples) == len(REPORTING_CASES)
    assert examples[0]["outputs"]["required_facts"] == REPORTING_CASES[0].required_facts


def test_target_reporting_calls_draft_report(monkeypatch) -> None:
    monkeypatch.setattr(
        langsmith_eval, "draft_report", lambda objective, findings, results, llm: "# Report body"
    )

    result = langsmith_eval.target_reporting(
        {"objective": "x", "findings": [], "results": []}, llm=object()
    )

    assert result == {"report": "# Report body"}


def test_deterministic_evaluator_reporting_passes_when_facts_and_sections_present() -> None:
    run = _FakeRun(outputs={"report": "# Objective\nSummary.\n# Findings\nAcme is $49.\n# Analysis\nDone."})
    example = _FakeExample(outputs={"required_sections": ["Objective", "Findings"], "required_facts": ["49"]})

    result = langsmith_eval.deterministic_evaluator_reporting(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_reporting_fails_when_a_fact_is_missing() -> None:
    run = _FakeRun(outputs={"report": "# Objective\nSummary.\n# Findings\nNo numbers here."})
    example = _FakeExample(outputs={"required_sections": ["Objective"], "required_facts": ["49"]})

    result = langsmith_eval.deterministic_evaluator_reporting(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_reporting_returns_judge_score() -> None:
    run = _FakeRun(outputs={"report": "# Objective\nSummary."})
    example = _FakeExample(inputs={"findings": [], "results": []})
    fake_llm = _FakeStructuredJudgeLLM(score=0.7, passed=True, reasoning="faithful")

    result = langsmith_eval.llm_judge_reporting(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.7, "comment": "faithful"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k reporting`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to the import block:

```python
from market_research_team.agents.reporting.node import draft_report
from market_research_team.evaluation.golden_dataset import REPORTING_CASES
```

Append:

```python
def _reporting_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"objective": case.objective, "findings": case.findings, "results": case.results},
            "outputs": {
                "required_sections": case.required_sections,
                "required_facts": case.required_facts,
            },
        }
        for case in REPORTING_CASES
    ]


def target_reporting(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    report = draft_report(inputs["objective"], inputs["findings"], inputs["results"], llm)
    return {"report": report}


def deterministic_evaluator_reporting(run: Any, example: Any) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    expected = example.outputs or {}
    sections_passed, sections_detail = checks.check_contains_all(
        report, expected.get("required_sections", [])
    )
    facts_passed, facts_detail = checks.check_contains_all(report, expected.get("required_facts", []))
    passed = sections_passed and facts_passed
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"{sections_detail}; {facts_detail}",
    }


_REPORTING_JUDGE_PROMPT = (
    "You are grading a markdown research report for faithfulness to its given "
    "findings/results and overall structure/clarity. Score 1.0 only if the report "
    "is well-organized and introduces no facts beyond what was given."
)


def llm_judge_reporting(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    report = (run.outputs or {}).get("report", "")
    findings = (example.inputs or {}).get("findings", [])
    results = (example.inputs or {}).get("results", [])
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_REPORTING_JUDGE_PROMPT),
            HumanMessage(content=f"Findings: {findings}\nResults: {results}\nReport:\n{report}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add reporting LangSmith target/evaluators"
```

---

### Task 6: `full_pipeline` category

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: `graph_module.run_graph(initial_state) -> AgentState` (existing, `graph.py`, imported
  via `from market_research_team import graph as graph_module` — same indirection
  `offline_eval.py` already uses so tests can monkeypatch it), `FULL_PIPELINE_CASES` (existing,
  `evaluation/golden_dataset.py`).
- Produces (used by Task 7): `_full_pipeline_examples() -> list[dict[str, Any]]`,
  `target_full_pipeline(inputs, *, llm) -> dict[str, Any]`,
  `deterministic_evaluator_full_pipeline(run, example) -> dict[str, Any]`,
  `llm_judge_full_pipeline(run, example, *, llm) -> dict[str, Any]` (reads the report file from
  disk — the only judge that does, since this category's real output is a file, not an in-memory
  value).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_langsmith_eval.py`:

```python
def test_full_pipeline_examples_match_golden_dataset() -> None:
    from market_research_team.evaluation.golden_dataset import FULL_PIPELINE_CASES

    examples = langsmith_eval._full_pipeline_examples()

    assert len(examples) == len(FULL_PIPELINE_CASES)
    assert examples[0]["inputs"] == {"objective": FULL_PIPELINE_CASES[0].objective}
    assert examples[0]["outputs"] == {}


def test_target_full_pipeline_calls_run_graph(monkeypatch) -> None:
    def _fake_run_graph(state, **_kwargs):
        return {
            **state,
            "research_findings": [{"source": "x", "content": "y", "relevance_score": 0.9}],
            "analytics_results": [],
            "report_path": "reports/mock.md",
            "error": None,
        }

    monkeypatch.setattr(langsmith_eval.graph_module, "run_graph", _fake_run_graph)

    result = langsmith_eval.target_full_pipeline({"objective": "assess pricing"}, llm=object())

    assert result == {
        "error": None,
        "report_path": "reports/mock.md",
        "findings_count": 1,
        "results_count": 0,
    }


def test_deterministic_evaluator_full_pipeline_passes_on_clean_run() -> None:
    run = _FakeRun(outputs={"error": None, "report_path": "reports/mock.md"})
    example = _FakeExample(inputs={"objective": "x"})

    result = langsmith_eval.deterministic_evaluator_full_pipeline(run, example)  # type: ignore[arg-type]

    assert result["score"] == 1.0


def test_deterministic_evaluator_full_pipeline_fails_when_error_is_set() -> None:
    run = _FakeRun(outputs={"error": "boom", "report_path": None})
    example = _FakeExample(inputs={"objective": "x"})

    result = langsmith_eval.deterministic_evaluator_full_pipeline(run, example)  # type: ignore[arg-type]

    assert result["score"] == 0.0


def test_llm_judge_full_pipeline_reads_report_file_and_returns_judge_score(tmp_path) -> None:
    report_file = tmp_path / "mock.md"
    report_file.write_text("# Objective\nAcme vs Globex pricing.\n", encoding="utf-8")
    run = _FakeRun(outputs={"report_path": str(report_file)})
    example = _FakeExample(inputs={"objective": "Assess Acme vs Globex pricing"})
    fake_llm = _FakeStructuredJudgeLLM(score=0.95, passed=True, reasoning="on target")

    result = langsmith_eval.llm_judge_full_pipeline(run, example, llm=fake_llm)  # type: ignore[arg-type]

    assert result == {"key": "llm_judge", "score": 0.95, "comment": "on target"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k full_pipeline`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to the import block:

```python
from pathlib import Path

from market_research_team import graph as graph_module
from market_research_team.evaluation.golden_dataset import FULL_PIPELINE_CASES
```

Append:

```python
def _full_pipeline_examples() -> list[dict[str, Any]]:
    return [{"inputs": {"objective": case.objective}, "outputs": {}} for case in FULL_PIPELINE_CASES]


def target_full_pipeline(inputs: dict[str, Any], *, llm: BaseChatModel) -> dict[str, Any]:
    # llm accepted for signature uniformity with the other target_* functions (all bound via
    # functools.partial(fn, llm=llm) in run_langsmith_eval) -- run_graph builds its own LLMs
    # internally via get_chat_model(), so it's unused here.
    initial_state: AgentState = {
        "messages": [],
        "objective": inputs["objective"],
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }
    final_state = graph_module.run_graph(initial_state)
    return {
        "error": final_state.get("error"),
        "report_path": final_state.get("report_path"),
        "findings_count": len(final_state.get("research_findings", [])),
        "results_count": len(final_state.get("analytics_results", [])),
    }


def deterministic_evaluator_full_pipeline(run: Any, example: Any) -> dict[str, Any]:
    outputs = run.outputs or {}
    passed = outputs.get("error") is None and bool(outputs.get("report_path"))
    return {
        "key": "deterministic",
        "score": 1.0 if passed else 0.0,
        "comment": f"error={outputs.get('error')!r}, report_path={outputs.get('report_path')!r}",
    }


_FULL_PIPELINE_JUDGE_PROMPT = (
    "You are grading the overall quality of a market research report written by a "
    "multi-agent pipeline, given the original objective. Score 1.0 if the report "
    "substantively and coherently addresses the objective."
)


def llm_judge_full_pipeline(run: Any, example: Any, *, llm: BaseChatModel) -> dict[str, Any]:
    outputs = run.outputs or {}
    report_path = outputs.get("report_path")
    objective = (example.inputs or {}).get("objective", "")
    report_text = Path(report_path).read_text(encoding="utf-8") if report_path else ""
    structured_llm = llm.with_structured_output(_JudgeScoreSchema)
    judgment = structured_llm.invoke(
        [
            SystemMessage(content=_FULL_PIPELINE_JUDGE_PROMPT),
            HumanMessage(content=f"Objective: {objective}\nReport:\n{report_text}"),
        ]
    )
    return {"key": "llm_judge", "score": judgment.score, "comment": judgment.reasoning}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add full_pipeline LangSmith target/evaluators"
```

---

### Task 7: Orchestration — `run_langsmith_eval`

**Files:**
- Modify: `src/market_research_team/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Consumes: every `target_*` / `deterministic_evaluator_*` / `llm_judge_*` / `_*_examples`
  function from Tasks 2-6, `sync_dataset` (Task 1), `LangSmithEvalSummary` (Task 1),
  `get_chat_model()` (existing, `llm.py`), `settings` (existing, `config.py`).
- Produces (used by Task 8): `run_langsmith_eval(provider: Literal["anthropic", "openai"] | None =
  None) -> list[LangSmithEvalSummary]`.

- [ ] **Step 1: Write failing tests**

Add `import pytest` to the top of `tests/test_langsmith_eval.py`'s import block (needed for
`pytest.raises` below — nothing before this task in the file used it). Then append:

```python
def test_run_langsmith_eval_computes_pass_rate_and_restores_provider(monkeypatch) -> None:
    from market_research_team.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(langsmith_eval, "get_chat_model", lambda: object())
    monkeypatch.setattr(langsmith_eval, "Client", lambda: _FakeLangSmithClient())
    monkeypatch.setattr(
        langsmith_eval, "sync_dataset", lambda client, category, examples: _FakeDataset(category)
    )

    class _FakeEvalResult:
        def __init__(self, score: float) -> None:
            self.score = score

    class _FakeExperimentResults:
        def __init__(self, name: str, rows: list[dict]) -> None:
            self._name = name
            self._rows = rows

        def experiment_name(self) -> str:
            return self._name

        def __iter__(self):
            return iter(self._rows)

    def _fake_evaluate(_target, *, data, evaluators, experiment_prefix, client):
        # data/evaluators/client are unused here but must keep these exact names --
        # run_langsmith_eval calls evaluate(..., data=..., evaluators=..., client=...) by keyword.
        return _FakeExperimentResults(
            f"{experiment_prefix}-1",
            [{"evaluation_results": {"results": [_FakeEvalResult(1.0), _FakeEvalResult(1.0)]}}],
        )

    monkeypatch.setattr(langsmith_eval, "evaluate", _fake_evaluate)

    summaries = langsmith_eval.run_langsmith_eval("openai")

    assert len(summaries) == 5
    assert {summary.category for summary in summaries} == {
        "query_rewrite",
        "supervisor_decision",
        "analytics",
        "reporting",
        "full_pipeline",
    }
    assert all(summary.pass_rate == 1.0 for summary in summaries)
    assert settings.llm_provider == "anthropic"


def test_run_langsmith_eval_restores_provider_even_on_failure(monkeypatch) -> None:
    from market_research_team.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(langsmith_eval, "get_chat_model", _boom)

    with pytest.raises(RuntimeError):
        langsmith_eval.run_langsmith_eval("openai")

    assert settings.llm_provider == "anthropic"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_langsmith_eval.py -v -k run_langsmith_eval`
Expected: FAIL with `AttributeError: module ... has no attribute 'run_langsmith_eval'`.

- [ ] **Step 3: Implement**

Add to the import block:

```python
import functools
from typing import Literal

from langsmith import Client, evaluate

from market_research_team.config import settings
from market_research_team.llm import get_chat_model
```

Append (at the end of the file):

```python
_CATEGORY_SPECS: list[tuple[str, Any, Any, Any, Any]] = [
    (
        "query_rewrite",
        _query_rewrite_examples,
        target_query_rewrite,
        deterministic_evaluator_query_rewrite,
        llm_judge_query_rewrite,
    ),
    (
        "supervisor_decision",
        _supervisor_decision_examples,
        target_supervisor_decision,
        deterministic_evaluator_supervisor_decision,
        llm_judge_supervisor_decision,
    ),
    (
        "analytics",
        _analytics_examples,
        target_analytics,
        deterministic_evaluator_analytics,
        llm_judge_analytics,
    ),
    (
        "reporting",
        _reporting_examples,
        target_reporting,
        deterministic_evaluator_reporting,
        llm_judge_reporting,
    ),
    (
        "full_pipeline",
        _full_pipeline_examples,
        target_full_pipeline,
        deterministic_evaluator_full_pipeline,
        llm_judge_full_pipeline,
    ),
]


def run_langsmith_eval(provider: Literal["anthropic", "openai"] | None = None) -> list[LangSmithEvalSummary]:
    """Sync each category's dataset and run a LangSmith experiment against it, layering the
    existing deterministic checks with a category-tailored LLM-judge. Mirrors
    offline_eval.run_all()'s provider-override-then-restore pattern. Requires
    LANGSMITH_API_KEY to be set (validated by the caller -- see scripts/run_evals.py)."""

    original_provider = settings.llm_provider
    if provider is not None:
        settings.llm_provider = provider

    try:
        llm = get_chat_model()
        client = Client()
        summaries: list[LangSmithEvalSummary] = []
        for category, examples_fn, target_fn, deterministic_fn, judge_fn in _CATEGORY_SPECS:
            examples = examples_fn()
            dataset = sync_dataset(client, category, examples)
            results = evaluate(
                functools.partial(target_fn, llm=llm),
                data=dataset.name,
                evaluators=[deterministic_fn, functools.partial(judge_fn, llm=llm)],
                experiment_prefix=f"market-research-team-{category}",
                client=client,
            )
            rows = list(results)
            total = len(rows)
            fully_passed = sum(
                1
                for row in rows
                if all(result.score == 1.0 for result in row["evaluation_results"]["results"])
            )
            pass_rate = fully_passed / total if total else 0.0
            summaries.append(LangSmithEvalSummary(category, results.experiment_name(), pass_rate))
        return summaries
    finally:
        settings.llm_provider = original_provider
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_langsmith_eval.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/market_research_team/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): add run_langsmith_eval orchestration across all categories"
```

---

### Task 8: `--langsmith` flag on `scripts/run_evals.py`

**Files:**
- Modify: `scripts/run_evals.py`
- Test: `tests/test_run_evals_cli.py`

**Interfaces:**
- Consumes: `run_langsmith_eval` and `LangSmithEvalSummary` (Task 7,
  `evaluation/langsmith_eval.py`).
- Produces: `_langsmith_prereq_error(langsmith_flag: bool, api_key: str | None) -> str | None`
  (pure helper, unit-tested directly); `--langsmith` CLI flag behavior.

- [ ] **Step 1: Write failing test for the pure prereq-check helper**

Create `tests/test_run_evals_cli.py`:

```python
"""Unit tests for the pure helper in scripts/run_evals.py.

Loaded by file path since `scripts/` isn't a package, same pattern as
tests/test_setup_env.py.
"""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_evals.py"
_spec = importlib.util.spec_from_file_location("run_evals", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evals)


def test_langsmith_prereq_error_when_flag_set_and_key_missing() -> None:
    assert run_evals._langsmith_prereq_error(True, None) == (
        "--langsmith requires LANGSMITH_API_KEY to be set"
    )


def test_langsmith_prereq_error_none_when_flag_set_and_key_present() -> None:
    assert run_evals._langsmith_prereq_error(True, "sk-fake") is None


def test_langsmith_prereq_error_none_when_flag_not_set() -> None:
    assert run_evals._langsmith_prereq_error(False, None) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_evals_cli.py -v`
Expected: FAIL with `AttributeError: module 'run_evals' has no attribute
'_langsmith_prereq_error'`.

- [ ] **Step 3: Implement the flag in `scripts/run_evals.py`**

Modify `scripts/run_evals.py`. Add `import os` to the existing imports at the top (alongside the
existing `argparse`, `sys`, `Path` imports), and add:

```python
from market_research_team.evaluation.langsmith_eval import LangSmithEvalSummary, run_langsmith_eval
```

Add this pure helper function right after `_print_report`:

```python
def _langsmith_prereq_error(langsmith_flag: bool, api_key: str | None) -> str | None:
    if langsmith_flag and not api_key:
        return "--langsmith requires LANGSMITH_API_KEY to be set"
    return None


def _print_langsmith_report(summaries: list[LangSmithEvalSummary]) -> None:
    print("\n=== LangSmith experiments ===")
    for summary in summaries:
        print(
            f"  {summary.category}: pass_rate={summary.pass_rate:.2f} "
            f"experiment={summary.experiment_name!r}"
        )
```

Replace the body of `main()` with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run against both anthropic and openai and report both (needs both API keys set).",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help=(
            "Also sync golden-dataset examples to LangSmith and run LLM-judge experiments "
            "(requires LANGSMITH_API_KEY)."
        ),
    )
    args = parser.parse_args()

    prereq_error = _langsmith_prereq_error(args.langsmith, os.environ.get("LANGSMITH_API_KEY"))
    if prereq_error is not None:
        print(prereq_error, file=sys.stderr)
        sys.exit(1)

    all_results: list[EvalResult] = []
    if args.compare:
        for provider in ("anthropic", "openai"):
            all_results += run_all(provider)  # type: ignore[arg-type]
    else:
        all_results += run_all(args.provider)  # type: ignore[arg-type]

    _print_report(all_results)
    output_path = save_results(all_results, _RESULTS_DIR)
    print(f"\nSaved results to {output_path}")

    langsmith_summaries: list[LangSmithEvalSummary] = []
    if args.langsmith:
        if args.compare:
            for provider in ("anthropic", "openai"):
                langsmith_summaries += run_langsmith_eval(provider)
        else:
            langsmith_summaries += run_langsmith_eval(args.provider)
        _print_langsmith_report(langsmith_summaries)

    if any(not result.passed for result in all_results) or any(
        summary.pass_rate < 1.0 for summary in langsmith_summaries
    ):
        print("\nRegression suite FAILED — one or more cases did not pass.")
        sys.exit(1)

    print("\nAll eval cases passed.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_evals_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full hermetic suite to confirm nothing else broke**

Run: `pytest -v`
Expected: all tests pass, including the pre-existing `tests/test_eval_offline.py` and
`tests/test_eval_checks.py` (unaffected by this change) and the new
`tests/test_langsmith_eval.py` / `tests/test_run_evals_cli.py`.

- [ ] **Step 6: Lint**

Run: `ruff check src tests scripts`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_evals.py tests/test_run_evals_cli.py
git commit -m "feat(eval): add --langsmith flag to scripts/run_evals.py"
```

---

### Task 9: Restore CI, update docs, final verification

**Files:**
- Modify: `.gitlab-ci.yml`
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:** None (documentation and CI config only — no code interfaces).

- [ ] **Step 1: Restore `.gitlab-ci.yml`**

Replace the (empty) contents of `.gitlab-ci.yml` with its pre-`3d08014` content:

```yaml
stages:
  - test

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  key: "$CI_COMMIT_REF_SLUG"
  paths:
    - .cache/pip

test:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install --upgrade pip
    - pip install -e ".[dev]"
  script:
    - ruff check src tests scripts
    - pytest -q
```

- [ ] **Step 2: Update `.env.example`**

In `.env.example`, change the `# LangGraph Studio / LangSmith tracing` block from:

```
# LangGraph Studio / LangSmith tracing (optional but recommended)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=market-research-analyst-team
```

to:

```
# LangGraph Studio / LangSmith tracing (optional but recommended)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=market-research-analyst-team

# Required for `python scripts/run_evals.py --langsmith` (dataset sync + LLM-judge
# experiments). Can be the same value as LANGCHAIN_API_KEY above.
LANGSMITH_API_KEY=
```

- [ ] **Step 3: Update README's configuration table**

In `README.md`, replace this table row:

```
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | off | Optional LangSmith tracing |
```

with:

```
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | off | Optional LangSmith tracing |
| `LANGSMITH_API_KEY` | unset | Required for `scripts/run_evals.py --langsmith` (dataset sync + LLM-judge experiments) |
```

- [ ] **Step 4: Update README's "Prompt evaluation regression suite" section**

In `README.md`, replace:

```
Run it after changing a system prompt, switching models, or before a
release — not on every commit, since it costs real API calls:

```bash
python scripts/run_evals.py                  # current LLM_PROVIDER
python scripts/run_evals.py --provider openai
python scripts/run_evals.py --compare         # anthropic AND openai, side by side
```

Exits non-zero if any case fails, and writes a timestamped JSON report to
`data/eval_results/`. The full-pipeline case needs the vector store seeded
(step 3 above). See `src/market_research_team/evaluation/golden_dataset.py`
to add cases.
```

with:

```
Run it after changing a system prompt, switching models, or before a
release — not on every commit, since it costs real API calls:

```bash
python scripts/run_evals.py                  # current LLM_PROVIDER
python scripts/run_evals.py --provider openai
python scripts/run_evals.py --compare         # anthropic AND openai, side by side
python scripts/run_evals.py --langsmith       # also run LangSmith dataset sync + LLM-judge experiments
```

Exits non-zero if any case fails, and writes a timestamped JSON report to
`data/eval_results/`. The full-pipeline case needs the vector store seeded
(step 3 above). See `src/market_research_team/evaluation/golden_dataset.py`
to add cases.

`--langsmith` requires `LANGSMITH_API_KEY` (see [Configuration](#configuration)). It layers
LLM-judge scoring on top of — not instead of — the deterministic checks above: each of the 5
categories gets its cases mirrored into a LangSmith Dataset
(`market-research-team-<category>`) and scored in a LangSmith Experiment by both the existing
deterministic check and a category-tailored LLM judge (relevance / groundedness / appropriateness
depending on category). See `src/market_research_team/evaluation/langsmith_eval.py`.
```

- [ ] **Step 5: Update README's "Known limitations" section**

In `README.md`, remove this bullet entirely from the "Known limitations" list:

```
- **CI is currently disabled.** `.gitlab-ci.yml` was added in `fdde9ff` (ran `ruff` + `pytest` on
  every push) but was later commented out and then emptied (`3d08014`, `97f7b2f`). No automated
  gate currently runs on push — needs a decision: restore it or remove the dead file.
```

- [ ] **Step 6: Fix the now-stale historical table row**

In `README.md`'s "Post-sprint — Hardening & productionization" table, change:

```
| GitLab CI: automated `ruff` + `pytest` on every push (currently disabled — see [Known limitations](#known-limitations)) |
```

to:

```
| GitLab CI: automated `ruff` + `pytest` on every push (restored) |
```

- [ ] **Step 7: Run the full verification suite**

Run: `ruff check src tests scripts && pytest -q`
Expected: `ruff` clean; all tests pass (the original 103 plus the new ones added in Tasks 1-8).

- [ ] **Step 8: Commit**

```bash
git add .gitlab-ci.yml .env.example README.md
git commit -m "chore: restore CI and document the --langsmith eval flag"
```

---

## Post-plan note

This plan covers sub-project 1 of 6 identified during scoping (see the spec's "Scope" section).
The remaining five — human-in-the-loop approval, MCP architecture improvements, security
hardening, a streaming UI, and any further best-practices findings — are separate sub-projects,
each requiring its own brainstorming → spec → plan cycle before implementation.
