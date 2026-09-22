"""Cross-encoder reranking for the Research Agent retrieval pipeline."""

from typing import Protocol

from langchain_core.documents import Document


class CrossEncoderModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def load_cross_encoder(model_name: str) -> CrossEncoderModel:
    """Load the sentence-transformers cross-encoder used to rerank candidates."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str,
    documents: list[Document],
    model: CrossEncoderModel,
    *,
    top_n: int = 5,
    score_floor: float | None = None,
) -> list[tuple[Document, float]]:
    """Rescore retrieved documents against the query with a cross-encoder.

    Cross-encoders score a (query, document) pair jointly rather than
    comparing independent embeddings, which is far more precise than the
    bi-encoder similarity search used for initial retrieval — but too slow
    to run over a whole corpus, so it's only applied to the already
    narrowed-down candidate set. Returns up to `top_n` documents sorted
    best-first, each paired with its cross-encoder score, trimming context
    down to what's actually worth spending the LLM's context window on.

    `score_floor` is a retrieval guardrail: candidates scoring below it are
    dropped even if fewer than `top_n` remain, so an off-topic objective
    yields an empty context instead of the least-bad chunks.
    """

    if not documents:
        return []

    pairs = [(query, document.page_content) for document in documents]
    scores = [float(score) for score in model.predict(pairs)]

    scored = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    if score_floor is not None:
        scored = [pair for pair in scored if pair[1] >= score_floor]
    return scored[:top_n]
