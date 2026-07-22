"""Hierarchical + recursive chunking for ingested documents.

Two-level strategy: split each document into markdown-header-delimited
"section" chunks (the parent level, used to expand context after
retrieval), then recursively split each section into embedding-sized
"leaf" chunks (the level actually indexed for vector search). Plain text
without headers falls back to a single implicit section.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

ChunkLevel = Literal["section", "leaf"]


@dataclass
class HierarchicalChunk:
    id: str
    doc_id: str
    parent_id: str | None
    level: ChunkLevel
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _doc_id(document: Document) -> str:
    source = str(document.metadata.get("source", ""))
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"doc-{digest}"


def chunk_document(
    document: Document,
    *,
    section_chunk_size: int = 2000,
    leaf_chunk_size: int = 800,
    leaf_chunk_overlap: int = 120,
) -> list[HierarchicalChunk]:
    """Split one loaded document into section (parent) and leaf (child) chunks."""

    doc_id = _doc_id(document)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    sections = header_splitter.split_text(document.page_content)
    if not sections:
        sections = [Document(page_content=document.page_content, metadata={})]

    # Re-split any section that's still larger than section_chunk_size, so no
    # single "parent" chunk blows past what we'd want to inject as context.
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=section_chunk_size, chunk_overlap=0
    )
    leaf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=leaf_chunk_size, chunk_overlap=leaf_chunk_overlap
    )

    chunks: list[HierarchicalChunk] = []
    section_index = 0
    for section in sections:
        section_texts = section_splitter.split_text(section.page_content) or [
            section.page_content
        ]
        for section_text in section_texts:
            section_id = f"{doc_id}::s{section_index}"
            section_index += 1
            section_metadata = {**document.metadata, **section.metadata, "doc_id": doc_id}
            chunks.append(
                HierarchicalChunk(
                    id=section_id,
                    doc_id=doc_id,
                    parent_id=None,
                    level="section",
                    content=section_text,
                    metadata=section_metadata,
                )
            )

            leaf_texts = leaf_splitter.split_text(section_text)
            for leaf_index, leaf_text in enumerate(leaf_texts):
                chunks.append(
                    HierarchicalChunk(
                        id=f"{section_id}::c{leaf_index}",
                        doc_id=doc_id,
                        parent_id=section_id,
                        level="leaf",
                        content=leaf_text,
                        metadata=section_metadata,
                    )
                )

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    section_chunk_size: int = 2000,
    leaf_chunk_size: int = 800,
    leaf_chunk_overlap: int = 120,
) -> list[HierarchicalChunk]:
    all_chunks: list[HierarchicalChunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                section_chunk_size=section_chunk_size,
                leaf_chunk_size=leaf_chunk_size,
                leaf_chunk_overlap=leaf_chunk_overlap,
            )
        )
    return all_chunks
