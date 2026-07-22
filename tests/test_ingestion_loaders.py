from pathlib import Path

from market_research_team.ingestion.loaders import load_raw_documents


def test_load_raw_documents_reads_markdown_and_text(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Title\n\nSome content.", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("Just plain text.", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    documents = load_raw_documents(tmp_path)

    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {"notes.md", "plain.txt"}
    assert len(documents) == 2


def test_load_raw_documents_recurses_into_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "doc.md").write_text("# Nested\n\nContent.", encoding="utf-8")

    documents = load_raw_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == str(Path("nested") / "doc.md")
