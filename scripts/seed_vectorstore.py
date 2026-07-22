"""One-shot CLI: load data/raw, chunk hierarchically, and seed the vector store."""

from market_research_team.config import settings
from market_research_team.ingestion.chunking import chunk_documents
from market_research_team.ingestion.index_build import build_vectorstore, persist_parent_store
from market_research_team.ingestion.loaders import load_raw_documents


def main() -> None:
    documents = load_raw_documents(settings.raw_dir)
    if not documents:
        raise SystemExit(f"No documents found under {settings.raw_dir}")

    chunks = chunk_documents(
        documents,
        section_chunk_size=settings.section_chunk_size,
        leaf_chunk_size=settings.leaf_chunk_size,
        leaf_chunk_overlap=settings.leaf_chunk_overlap,
    )
    section_count = sum(1 for chunk in chunks if chunk.level == "section")
    leaf_count = sum(1 for chunk in chunks if chunk.level == "leaf")
    print(
        f"Loaded {len(documents)} documents -> "
        f"{section_count} sections, {leaf_count} leaf chunks"
    )

    parent_store_path = settings.processed_dir / "parent_sections.json"
    persist_parent_store(chunks, path=parent_store_path)
    print(f"Parent sections persisted to {parent_store_path}")

    build_vectorstore(
        chunks,
        persist_dir=settings.vectorstore_dir,
        embedding_model_name=settings.embedding_model_name,
    )
    print(f"Vector store persisted to {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()
