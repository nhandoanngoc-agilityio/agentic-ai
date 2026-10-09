"""Research node: targeted re-research on a hand-back, merging passes, and
ending runs that can't find anything instead of looping."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from market_research_team.agents.research import node as research_node_module
from market_research_team.agents.research.node import merge_findings, research_node
from market_research_team.config import settings
from market_research_team.state import AgentState, ResearchFinding


def _finding(content: str, score: float, source: str = "acme.md") -> ResearchFinding:
    return {"source": source, "content": content, "relevance_score": score}


def _state(findings: list[ResearchFinding], focus: str | None = None) -> AgentState:
    state: AgentState = {
        "messages": [],
        "objective": "Assess competitor pricing strategy",
        "next": "research",
        "research_findings": findings,
        "analytics_results": [],
        "report_path": None,
    }
    if focus is not None:
        state["research_focus"] = focus
    return state


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, findings: list[ResearchFinding]):
    calls: list[dict[str, Any]] = []

    def _pipeline(objective: str, focus: str | None = None, exclude=frozenset()):
        calls.append({"objective": objective, "focus": focus, "exclude": exclude})
        # Like the real pipeline: chunks already held are excluded.
        return [f for f in findings if (f["source"], f["content"]) not in exclude], 3, 9, []

    monkeypatch.setattr(research_node_module, "run_research_pipeline", _pipeline)
    return calls


def test_merge_findings_dedupes_sorts_and_counts_new_entries() -> None:
    old = [_finding("a", 1.0), _finding("b", 0.5)]
    new = [_finding("b", 0.9), _finding("c", 2.0)]

    merged, added = merge_findings(old, new)

    assert [f["content"] for f in merged] == ["c", "a", "b"]
    assert merged[2]["relevance_score"] == 0.9  # the better score of a duplicate wins
    assert added == 1


def test_merge_findings_caps_the_total_and_drops_lowest_scores() -> None:
    cap = settings.run_policy.max_findings
    old = [_finding(f"old{i}", float(i)) for i in range(cap)]

    merged, added = merge_findings(old, [_finding("best", 100.0), _finding("worst", -5.0)])

    assert len(merged) == cap
    assert merged[0]["content"] == "best"
    assert "worst" not in {f["content"] for f in merged}
    assert added == 1


def test_first_pass_ignores_a_stale_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_pipeline(monkeypatch, [_finding("a", 1.0)])

    update = research_node(_state([], focus="leftover"))

    assert calls == [
        {"objective": "Assess competitor pricing strategy", "focus": None, "exclude": frozenset()}
    ]
    assert update["research_findings"] == [_finding("a", 1.0)]
    assert "research_exhausted" not in update


def test_hand_back_targets_the_gap_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_pipeline(monkeypatch, [_finding("churn data", 3.0, "globex.md")])

    update = research_node(_state([_finding("a", 1.0)], focus="Globex churn"))

    assert calls[0]["focus"] == "Globex churn"
    assert [f["content"] for f in update["research_findings"]] == ["churn data", "a"]
    assert update["research_focus"] is None
    assert "research_exhausted" not in update
    assert "1 new (gap: Globex churn)" in update["messages"][0].content


def test_hand_back_that_adds_nothing_marks_research_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = [_finding("a", 1.0)]
    _stub_pipeline(monkeypatch, list(existing))

    update = research_node(_state(existing, focus="anything"))

    assert update["research_findings"] == existing
    assert update["research_exhausted"] is True


def test_empty_first_pass_ends_the_run_with_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_pipeline(monkeypatch, [])

    update = research_node(_state([]))

    assert "no relevant material" in update["error"]
    assert isinstance(update["messages"][0], AIMessage)


def test_targeted_hand_back_that_adds_nothing_marks_only_that_item_unanswerable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = [_finding("a", 1.0)]
    _stub_pipeline(monkeypatch, existing)
    state = _state(existing, focus="Globex churn")
    state["research_focus_id"] = "q2"
    state["plan"] = [
        {"id": "q1", "question": "Acme price?", "status": "answered", "sources": ["acme.md"]},
        {"id": "q2", "question": "Globex churn?", "status": "open", "sources": []},
        {"id": "q3", "question": "Market size?", "status": "open", "sources": []},
    ]

    update = research_node(state)

    assert [item["status"] for item in update["plan"]] == ["answered", "unanswerable", "open"]
    assert "research_exhausted" not in update
    assert "q2 unanswerable" in update["messages"][0].content
    assert update["research_focus_id"] is None


def test_targeted_hand_back_marks_its_item_searched_even_when_it_adds_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pipeline(monkeypatch, [_finding("new globex fact", 2.0, source="globex.md")])
    state = _state([_finding("a", 1.0)], focus="Globex ACV?")
    state["research_focus_id"] = "q2"
    state["plan"] = [
        {"id": "q1", "question": "Acme price?", "status": "open", "sources": []},
        {"id": "q2", "question": "Globex ACV?", "status": "open", "sources": []},
    ]

    update = research_node(state)

    q1, q2 = update["plan"]
    assert q2 == {**state["plan"][1], "attempted": True}  # still open: the supervisor judges it
    assert "attempted" not in q1


def test_hand_back_excludes_held_chunks_from_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_pipeline(monkeypatch, [_finding("market size", 2.0, "market.md")])
    held = [_finding("a", 1.0), _finding("b", 0.5, "globex.md")]

    research_node(_state(held, focus="Market size?"))

    assert calls[0]["exclude"] == frozenset({("acme.md", "a"), ("globex.md", "b")})


def test_a_targeted_pass_keeps_its_findings_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scores from different passes are ranked against different queries: a
    gap chunk scoring below every held finding is still kept, and the
    lowest-scoring held finding makes room."""

    held = [_finding(f"pricing {i}", 5.0 + i) for i in range(10)]
    _stub_pipeline(monkeypatch, [_finding("market size and growth", -2.0, "market.md")])
    state = _state(held, focus="Market size?")
    state["research_focus_id"] = "q4"
    state["plan"] = [{"id": "q4", "question": "Market size?", "status": "open", "sources": []}]

    update = research_node(state)

    contents = [f["content"] for f in update["research_findings"]]
    assert len(contents) == 10
    assert "market size and growth" in contents
    assert "pricing 0" not in contents  # the lowest-scoring held finding made room
    assert update["plan"][0]["status"] == "open"  # found something: not unanswerable


def test_merge_findings_keep_new_never_drops_the_new_pass() -> None:
    from market_research_team.agents.research.node import merge_findings

    existing = [_finding(f"e{i}", 9.0) for i in range(10)]
    new = [_finding("n1", -1.0), _finding("n2", -3.0)]

    merged, added = merge_findings(existing, new, keep_new=True)

    assert added == 2
    assert {f["content"] for f in merged} >= {"n1", "n2"}
    assert len(merged) == 10


def test_pipeline_drops_held_chunks_before_reranking_against_the_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline itself, with only its collaborators stubbed: held chunks
    never reach the reranker, and a targeted pass is ranked against the gap
    rather than the (pricing) objective."""

    from langchain_core.documents import Document

    held = Document(page_content="Acme is $49 per seat.", metadata={"source": "acme.md"})
    market = Document(page_content="The BI market grows 11% a year.", metadata={"source": "m.md"})
    seen: dict[str, Any] = {}

    def _rerank(query, candidates, *_args, **_kwargs):
        seen["query"], seen["candidates"] = query, list(candidates)
        return [(document, 1.0) for document in candidates]

    module = research_node_module
    monkeypatch.setattr(module, "get_chat_model", lambda: None)
    monkeypatch.setattr(module, "rewrite_and_expand", lambda objective, llm, focus=None: ["q"])
    monkeypatch.setattr(module, "load_vectorstore", lambda **_kwargs: None)
    monkeypatch.setattr(module, "retrieve_for_queries_cached", lambda *a, **k: [held, market])
    monkeypatch.setattr(module, "cached_cross_encoder", lambda name: None)
    monkeypatch.setattr(module, "rerank_cached", _rerank)

    findings, _queries, candidate_count, _events = module.run_research_pipeline(
        "Compare Acme and Globex pricing",
        focus="What is the market size and growth?",
        exclude=frozenset({("acme.md", "Acme is $49 per seat.")}),
    )

    assert seen["query"] == "What is the market size and growth?"
    assert [d.page_content for d in seen["candidates"]] == ["The BI market grows 11% a year."]
    assert [f["source"] for f in findings] == ["m.md"]
    assert candidate_count == 1


def test_findings_carry_the_chunk_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document

    dated = Document(
        page_content="Acme Starter is $55.",
        metadata={
            "source": "bench.md",
            "entity": "Acme",
            "topic": "pricing",
            "as_of": "2026-08",
            "doc_type": "benchmark",
        },
    )
    plain = Document(page_content="Undated.", metadata={"source": "old.md"})
    module = research_node_module
    monkeypatch.setattr(module, "get_chat_model", lambda: None)
    monkeypatch.setattr(module, "rewrite_and_expand", lambda objective, llm, focus=None: ["q"])
    monkeypatch.setattr(module, "load_vectorstore", lambda **_kwargs: None)
    monkeypatch.setattr(module, "retrieve_for_queries_cached", lambda *a, **k: [dated, plain])
    monkeypatch.setattr(module, "cached_cross_encoder", lambda name: None)
    monkeypatch.setattr(
        module, "rerank_cached", lambda query, candidates, *a, **k: [(d, 1.0) for d in candidates]
    )

    findings, *_ = module.run_research_pipeline("Acme pricing")

    assert findings[0] == {
        "source": "bench.md",
        "content": "Acme Starter is $55.",
        "relevance_score": 1.0,
        "entity": "Acme",
        "topic": "pricing",
        "as_of": "2026-08",
        "doc_type": "benchmark",
    }
    assert findings[1] == {"source": "old.md", "content": "Undated.", "relevance_score": 1.0}
