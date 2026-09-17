"""Vector-store retrieval loop: runs a batch of queries and merges results."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_vectorstore(
    *,
    persist_dir: Path,
    embedding_model_name: str,
    collection_name: str = "market_research",
) -> Chroma:
    """Open the Chroma collection persisted by `ingestion.index_build`.

    Imports `langchain_huggingface` lazily so this module (and the query
    rewriting/retrieval logic below) stays importable and unit-testable
    without pulling in the sentence-transformers stack until a real
    embedding model is actually needed.
    """

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def retrieve_for_queries(
    vectorstore: Chroma,
    queries: list[str],
    *,
    k: int = 4,
) -> list[Document]:
    """Run similarity search for each query and merge the results.

    Duplicate chunks retrieved by more than one rewritten query are merged,
    keeping the best (lowest-distance) score seen across queries. Results
    are returned sorted best-first. Cross-encoder reranking on top of this
    merged set is layered on separately.
    """

    best_by_id: dict[str, tuple[float, Document]] = {}
    for query in queries:
        for document, distance in vectorstore.similarity_search_with_score(query, k=k):
            chunk_id = document.metadata.get("chunk_id", document.page_content)
            existing = best_by_id.get(chunk_id)
            if existing is None or distance < existing[0]:
                best_by_id[chunk_id] = (distance, document)

    ranked = sorted(best_by_id.values(), key=lambda pair: pair[0])
    merged: list[Document] = []
    for distance, document in ranked:
        document.metadata = {**document.metadata, "retrieval_distance": distance}
        merged.append(document)
    return merged
