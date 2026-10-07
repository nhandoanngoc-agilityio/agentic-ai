"""Research Agent node: query rewriting -> vector retrieval -> cross-encoder reranking."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from market_research_team.caching.memo import cached_cross_encoder
from market_research_team.caching.rerank_cache import rerank_cached
from market_research_team.caching.retrieval_cache import retrieve_for_queries_cached
from market_research_team.config import settings
from market_research_team.llm import get_chat_model
from market_research_team.retrieval.query_rewriter import rewrite_and_expand
from market_research_team.retrieval.retriever import load_vectorstore
from market_research_team.security.patterns import INJECTION_PATTERNS, find_matches
from market_research_team.state import AgentState, GuardrailEvent, ResearchFinding

_RETRIEVAL_K_PER_QUERY = 4
_RERANK_TOP_N = 5
# Cap on findings kept across research passes (each pass adds up to
# `_RERANK_TOP_N`), so repeated hand-backs can't grow the analytics and
# report prompts without bound. Lowest-scoring findings are dropped first.
_MAX_FINDINGS = 10


def filter_injected_chunks(
    reranked: list[tuple[Document, float]],
) -> tuple[list[tuple[Document, float]], list[GuardrailEvent]]:
    """Retrieval guardrail: drop chunks that contain prompt-injection text.

    The ingestion pipeline accepts any file placed in `data/raw/`, so a
    poisoned document is the realistic way instructions reach the report
    prompt. The reporting prompt already fences chunks as data; this removes
    the obvious cases before they get that far, and records each drop.
    """

    kept: list[tuple[Document, float]] = []
    events: list[GuardrailEvent] = []
    for document, score in reranked:
        hits = find_matches(document.page_content, INJECTION_PATTERNS)
        if hits:
            source = document.metadata.get("source", "unknown")
            events.append(
                {
                    "layer": "retrieval",
                    "rule": "injection_in_chunk",
                    "detail": f"dropped chunk from {source}: {', '.join(hits)}",
                }
            )
            continue
        kept.append((document, score))
    return kept, events


def run_research_pipeline(
    objective: str,
    focus: str | None = None,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[list[ResearchFinding], int, int, list[GuardrailEvent]]:
    """Rewrite the objective into search queries, retrieve, then rerank.

    `focus` is the gap the supervisor named on a hand-back: queries target it
    and reranking scores chunks against the gap alone -- scored against the
    objective too, a pricing objective outranked a market-size gap's chunks.
    `exclude` holds the (source, content) of findings already held: they are
    dropped before reranking, so the top slots go to chunks that are new.
    Returns the reranked findings plus the query and candidate counts and any
    retrieval guardrail events, so the node can report them without
    recomputing anything.
    """

    llm = get_chat_model()
    queries = rewrite_and_expand(objective, llm, focus=focus)
    rerank_query = focus or objective

    vectorstore = load_vectorstore(
        persist_dir=settings.vectorstore_dir,
        embedding_model_name=settings.embedding_model_name,
    )
    candidates = retrieve_for_queries_cached(
        vectorstore,
        queries,
        k=_RETRIEVAL_K_PER_QUERY,
        embedding_model_name=settings.embedding_model_name,
        persist_dir=settings.vectorstore_dir,
    )
    candidates = [
        document
        for document in candidates
        if (document.metadata.get("source", "unknown"), document.page_content) not in exclude
    ]

    cross_encoder = cached_cross_encoder(settings.reranker_model_name)
    reranked = rerank_cached(
        rerank_query,
        candidates,
        cross_encoder,
        top_n=_RERANK_TOP_N,
        score_floor=settings.rerank_score_floor,
        reranker_model_name=settings.reranker_model_name,
        persist_dir=settings.vectorstore_dir,
    )
    reranked, events = filter_injected_chunks(reranked)

    findings: list[ResearchFinding] = [
        {
            "source": document.metadata.get("source", "unknown"),
            "content": document.page_content,
            "relevance_score": score,
        }
        for document, score in reranked
    ]
    return findings, len(queries), len(candidates), events


def _finding_key(finding: ResearchFinding) -> tuple[str, str]:
    return finding["source"], finding["content"]


def merge_findings(
    existing: list[ResearchFinding], new: list[ResearchFinding], *, keep_new: bool = False
) -> tuple[list[ResearchFinding], int]:
    """Union of two research passes, deduplicated by (source, content), best
    score first, capped at `_MAX_FINDINGS`. Returns the merged list and how
    many of its entries the new pass contributed.

    `keep_new` (a targeted pass): the new findings are kept and the
    lowest-scoring existing ones make room. Scores from different passes are
    ranked against different queries, so comparing them could drop exactly
    the gap-filling chunks the pass was sent for.
    """

    by_key = {_finding_key(finding): finding for finding in existing}
    seen = set(by_key)
    for finding in new:
        key = _finding_key(finding)
        if key not in by_key or finding["relevance_score"] > by_key[key]["relevance_score"]:
            by_key[key] = finding
    merged = sorted(by_key.values(), key=lambda f: f["relevance_score"], reverse=True)
    if keep_new:
        # New findings first, then the best existing ones fill what is left.
        new_keys = {_finding_key(finding) for finding in new}
        ordered = [f for f in merged if _finding_key(f) in new_keys] + [
            f for f in merged if _finding_key(f) not in new_keys
        ]
        kept = {_finding_key(f) for f in ordered[:_MAX_FINDINGS]}
        merged = [f for f in merged if _finding_key(f) in kept]
    merged = merged[:_MAX_FINDINGS]
    added = sum(1 for finding in merged if _finding_key(finding) not in seen)
    return merged, added


def research_node(state: AgentState) -> dict[str, Any]:
    existing = state.get("research_findings", [])
    # Only a hand-back (findings already exist) is targeted; a first pass
    # always searches the objective broadly.
    focus = state.get("research_focus") if existing else None
    if existing:
        held = frozenset(_finding_key(finding) for finding in existing)
        findings, query_count, candidate_count, events = run_research_pipeline(
            state["objective"], focus=focus, exclude=held
        )
    else:
        findings, query_count, candidate_count, events = run_research_pipeline(
            state["objective"], focus=focus
        )
    merged, added = merge_findings(existing, findings, keep_new=bool(existing))
    if not merged:
        # Nothing in the index cleared the rerank floor: re-running Research
        # would retrieve the same nothing until the recursion limit, so end
        # the run with a clear reason instead.
        message = "Research found no relevant material for this objective in the knowledge base."
        return {
            "error": message,
            "research_exhausted": True,
            "guardrail_events": events,
            "messages": [AIMessage(content=message, name="research_agent")],
        }

    update: dict[str, Any] = {
        "findings_version": state.get("findings_version", 0) + (1 if added else 0),
        "research_findings": merged,
        "research_focus": None,
        "research_focus_id": None,
        "guardrail_events": events,
    }
    summary = (
        f"Research complete: {query_count} queries, {candidate_count} candidates retrieved, "
        f"{len(findings)} kept after reranking"
    )
    if existing:
        summary += f", {added} new (gap: {focus or 'none given'})"
        focus_id = state.get("research_focus_id")
        plan = state.get("plan") or []
        targeted = bool(focus_id) and any(item["id"] == focus_id for item in plan)
        if targeted:
            # One targeted search per sub-question. Held chunks were excluded,
            # so an empty pass means no relevant new chunk cleared the rerank
            # floor: the knowledge base doesn't hold the answer.
            status_update = {"status": "unanswerable"} if not findings else {}
            update["plan"] = [
                {**item, "attempted": True, **status_update} if item["id"] == focus_id else item
                for item in plan
            ]
            if not findings:
                summary += f" -> {focus_id} unanswerable from the knowledge base"
        if added == 0 and not targeted:
            # Another pass would search the same index for the same thing:
            # tell the supervisor to stop offering Research.
            update["research_exhausted"] = True
    update["messages"] = [AIMessage(content=f"{summary}.", name="research_agent")]
    return update
