from langchain_core.documents import Document

from market_research_team.ingestion.chunking import chunk_document, chunk_documents

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


def test_chunks_carry_topic_and_the_resolved_entity() -> None:
    profile = Document(
        page_content="# Acme\n\n## Product and Pricing\n\nStarter $49.\n",
        metadata={"source": "acme.md", "entity": "Acme", "doc_type": "competitor_profile"},
    )
    bench = Document(
        page_content="# Benchmark\n\n## Per-Seat Pricing\n\n### Acme\n\nStarter $55.\n",
        metadata={"source": "bench.md", "entity": "Multiple", "doc_type": "benchmark"},
    )

    chunks = chunk_documents([profile, bench])

    priced = [c.metadata for c in chunks if c.level == "leaf" and "$" in c.content]
    assert {(m["source"], m["entity"], m["topic"]) for m in priced} == {
        ("acme.md", "Acme", "pricing"),
        ("bench.md", "Acme", "pricing"),
    }


def test_undated_documents_get_a_topic_but_no_entity() -> None:
    document = Document(page_content="# X\n\n## Strengths\n\nGood.\n", metadata={"source": "x.md"})

    leaf = next(c for c in chunk_documents([document]) if c.level == "leaf" and "Good" in c.content)

    assert leaf.metadata["topic"] == "strengths"
    assert "entity" not in leaf.metadata and "as_of" not in leaf.metadata
