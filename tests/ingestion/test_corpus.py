"""The committed corpus in data/raw/: headers, sizes, and the two built-in
conflicts -- and no others. Reads files only; no index, no network."""

import re
from typing import Any

from market_research_team.config import settings
from market_research_team.ingestion.chunking import chunk_documents
from market_research_team.ingestion.loaders import load_raw_documents
from market_research_team.retrieval.evidence import finding_key, superseded
from market_research_team.security.output_filters import (
    _NUMBER_TOKEN,
    _is_grounded,
    _token_value,
)

_TYPES = {
    "competitor_profile",
    "pricing_page",
    "benchmark",
    "analyst_note",
    "review_digest",
    "market_report",
}
_FIELDS = ("entity", "topic", "as_of", "doc_type")


def _documents() -> list[Any]:
    documents = load_raw_documents(settings.raw_dir)
    return [d for d in documents if d.metadata["source"].endswith(".md")]


def _leaf_findings() -> list[Any]:
    return [
        {
            "source": chunk.metadata["source"],
            "content": chunk.content,
            "relevance_score": 1.0,
            **{key: chunk.metadata[key] for key in _FIELDS if key in chunk.metadata},
        }
        for chunk in chunk_documents(_documents())
        if chunk.level == "leaf"
    ]


def _values(text: str) -> list[float]:
    return [_token_value(match) for match in _NUMBER_TOKEN.finditer(text)]


def test_every_document_has_a_valid_header_and_size() -> None:
    documents = _documents()

    assert len(documents) == 12
    for document in documents:
        source = document.metadata["source"]
        assert document.metadata.get("doc_type") in _TYPES, source
        assert re.fullmatch(r"\d{4}-\d{2}", document.metadata.get("as_of", "")), source
        assert document.metadata.get("entity"), source
        assert 300 <= len(document.page_content.split()) <= 450, source


def test_multi_vendor_documents_attribute_chunks_to_vendors() -> None:
    entities = {
        finding["entity"]
        for finding in _leaf_findings()
        if finding["source"] == "pricing_benchmark_2026.md"
    }

    assert {"Acme", "Initech", "Umbrella", "Hooli", "Vandelay"} <= entities


def test_only_the_two_built_in_conflicts_leave_stale_figures() -> None:
    findings = _leaf_findings()
    by = superseded(findings)
    current = {
        value for f in findings if finding_key(f) not in by for value in _values(f["content"])
    }
    stale_only = {
        (finding["entity"], value)
        for finding in findings
        if finding_key(finding) in by
        for value in _values(finding["content"])
        if value >= 10 and not _is_grounded(value, current)
    }

    assert stale_only == {("Acme", 49.0), ("Globex", 120_000.0), ("Globex", 350_000.0)}
