"""Embed leaf chunks into the vector store; persist section chunks as a parent docstore."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from market_research_team.ingestion.chunking import HierarchicalChunk


def build_vectorstore(
    chunks: list[HierarchicalChunk],
    *,
    persist_dir: Path,
    embedding_model_name: str,
    collection_name: str = "market_research",
) -> Chroma:
    """Embed leaf-level chunks and persist them to a local Chroma collection."""

    from langchain_huggingface import HuggingFaceEmbeddings

    leaf_chunks = [chunk for chunk in chunks if chunk.level == "leaf"]
    if not leaf_chunks:
        raise ValueError("No leaf-level chunks to index — check chunking output.")

    documents = [
        Document(
            page_content=chunk.content,
            metadata={**chunk.metadata, "chunk_id": chunk.id, "parent_id": chunk.parent_id},
        )
        for chunk in leaf_chunks
    ]
    ids = [chunk.id for chunk in leaf_chunks]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )


def persist_parent_store(chunks: list[HierarchicalChunk], *, path: Path) -> None:
    """Write section-level chunks to a JSON docstore keyed by chunk id.

    The Research Agent's retriever expands a matched leaf chunk back to its
    parent section via this file, giving the LLM more surrounding context
    than the leaf chunk alone provides.
    """

    sections = {
        chunk.id: {"content": chunk.content, "metadata": chunk.metadata}
        for chunk in chunks
        if chunk.level == "section"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
