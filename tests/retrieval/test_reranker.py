from langchain_core.documents import Document

from market_research_team.retrieval.reranker import rerank


class _FakeCrossEncoder:
    """Deterministic cross-encoder: scores by term overlap with the query."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, passage in pairs:
            query_terms = set(query.lower().split())
            passage_terms = set(passage.lower().split())
            scores.append(float(len(query_terms & passage_terms)))
        return scores


def _documents() -> list[Document]:
    return [
        Document(page_content="Acme pricing starts at forty nine dollars per seat."),
        Document(page_content="Globex has strong row level security."),
        Document(page_content="Acme onboarding is fast for new customers."),
    ]


def test_rerank_orders_by_cross_encoder_score_best_first() -> None:
    results = rerank("acme pricing", _documents(), _FakeCrossEncoder(), top_n=3)

    contents = [document.page_content for document, _score in results]
    assert contents[0] == "Acme pricing starts at forty nine dollars per seat."
    assert contents[-1] == "Globex has strong row level security."


def test_rerank_respects_top_n() -> None:
    results = rerank("acme", _documents(), _FakeCrossEncoder(), top_n=1)

    assert len(results) == 1


def test_rerank_returns_empty_list_for_no_candidates() -> None:
    results = rerank("acme", [], _FakeCrossEncoder(), top_n=5)

    assert results == []


def test_rerank_score_floor_drops_low_scoring_candidates() -> None:
    # Fake scores are term overlaps: "acme pricing" -> 2, 0, 1 for the three docs.
    results = rerank("acme pricing", _documents(), _FakeCrossEncoder(), top_n=3, score_floor=1.0)

    contents = [document.page_content for document, _score in results]
    assert contents == [
        "Acme pricing starts at forty nine dollars per seat.",
        "Acme onboarding is fast for new customers.",
    ]
    assert all(score >= 1.0 for _document, score in results)


def test_rerank_score_floor_can_return_nothing_for_off_topic_query() -> None:
    assert rerank("weather tomorrow", _documents(), _FakeCrossEncoder(), score_floor=1.0) == []


def test_rerank_without_floor_keeps_everything_up_to_top_n() -> None:
    assert len(rerank("weather tomorrow", _documents(), _FakeCrossEncoder(), top_n=3)) == 3
