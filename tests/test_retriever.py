from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from market_research_team.agents.research.retriever import retrieve_for_queries

_VOCABULARY = ["pricing", "security", "onboarding", "acme", "globex"]


class _FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding so retrieval is testable offline."""

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in _VOCABULARY]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build_vectorstore() -> Chroma:
    documents = [
        Document(
            page_content="Acme pricing starts at $49 per seat per month.",
            metadata={"chunk_id": "acme-pricing", "source": "competitor_acme.md"},
        ),
        Document(
            page_content="Globex offers strong row-level security for regulated industries.",
            metadata={"chunk_id": "globex-security", "source": "competitor_globex.md"},
        ),
        Document(
            page_content="Acme onboarding typically completes in under one week.",
            metadata={"chunk_id": "acme-onboarding", "source": "competitor_acme.md"},
        ),
    ]
    return Chroma.from_documents(
        documents=documents,
        embedding=_FakeEmbeddings(),
        ids=[doc.metadata["chunk_id"] for doc in documents],
    )


def test_retrieve_for_queries_merges_and_dedupes_across_queries() -> None:
    vectorstore = _build_vectorstore()

    results = retrieve_for_queries(vectorstore, ["acme pricing", "acme onboarding"], k=2)

    chunk_ids = [doc.metadata["chunk_id"] for doc in results]
    assert len(chunk_ids) == len(set(chunk_ids)), "duplicate chunks across queries must be merged"
    assert "acme-pricing" in chunk_ids
    assert "acme-onboarding" in chunk_ids


def test_retrieve_for_queries_ranks_best_match_first() -> None:
    vectorstore = _build_vectorstore()

    results = retrieve_for_queries(vectorstore, ["globex security"], k=3)

    assert results[0].metadata["chunk_id"] == "globex-security"


def test_retrieve_for_queries_annotates_retrieval_distance() -> None:
    vectorstore = _build_vectorstore()

    results = retrieve_for_queries(vectorstore, ["acme pricing"], k=1)

    assert "retrieval_distance" in results[0].metadata
