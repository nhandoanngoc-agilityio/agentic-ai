from langchain_core.documents import Document

from market_research_team.ingestion.chunking import chunk_document

_SAMPLE_MARKDOWN = """# Company Overview

Intro paragraph about the company.

## Product and Pricing

Pricing details go here.

### Strengths

Strength details go here.

### Weaknesses

Weakness details go here.
"""


def test_chunk_document_produces_sections_and_leaves_with_parent_links() -> None:
    document = Document(page_content=_SAMPLE_MARKDOWN, metadata={"source": "acme.md"})

    chunks = chunk_document(document, leaf_chunk_size=60, leaf_chunk_overlap=0)

    sections = [c for c in chunks if c.level == "section"]
    leaves = [c for c in chunks if c.level == "leaf"]

    assert len(sections) >= 3  # h1 preface + h2 + two h3 subsections
    assert leaves, "expected at least one leaf chunk"

    section_ids = {s.id for s in sections}
    for leaf in leaves:
        assert leaf.parent_id in section_ids
        assert leaf.doc_id == leaf.metadata["doc_id"]


def test_chunk_document_is_deterministic_across_calls() -> None:
    document = Document(page_content=_SAMPLE_MARKDOWN, metadata={"source": "acme.md"})

    first = chunk_document(document)
    second = chunk_document(document)

    assert [c.id for c in first] == [c.id for c in second]


def test_chunk_document_falls_back_to_single_section_without_headers() -> None:
    document = Document(
        page_content="Just a flat paragraph, no headers here.",
        metadata={"source": "flat.txt"},
    )

    chunks = chunk_document(document)

    sections = [c for c in chunks if c.level == "section"]
    assert len(sections) == 1
