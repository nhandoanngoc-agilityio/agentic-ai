from pathlib import Path

from market_research_team.ingestion.loaders import load_raw_documents, parse_front_matter


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


_HEADER = "---\nentity: Acme\ndoc_type: competitor_profile\nas_of: 2026-03\n---\n"


def test_front_matter_is_parsed_and_stripped() -> None:
    fields, body = parse_front_matter(_HEADER + "# Acme\n\nText.")

    assert fields == {"entity": "Acme", "doc_type": "competitor_profile", "as_of": "2026-03"}
    assert body == "# Acme\n\nText."


def test_text_without_front_matter_is_unchanged() -> None:
    assert parse_front_matter("# Acme\n\nText.") == ({}, "# Acme\n\nText.")


def test_an_unclosed_block_is_left_as_text() -> None:
    text = "---\nentity: Acme\n# Acme\n"
    assert parse_front_matter(text) == ({}, text)


def test_unknown_keys_and_a_malformed_date_are_dropped() -> None:
    fields, _ = parse_front_matter("---\nentity: Acme\nauthor: x\nas_of: March 2026\n---\nBody")

    assert fields == {"entity": "Acme"}


def test_windows_line_endings_and_a_bom_still_parse() -> None:
    text = "﻿" + _HEADER.replace("\n", "\r\n") + "# Acme\r\n"

    fields, body = parse_front_matter(text)

    assert fields["as_of"] == "2026-03"
    assert body == "# Acme\n"


def test_loaded_documents_carry_header_fields(tmp_path: Path) -> None:
    (tmp_path / "acme.md").write_text(_HEADER + "# Acme\n\nText.", encoding="utf-8")
    (tmp_path / "plain.md").write_text("# Plain\n", encoding="utf-8")

    acme, plain = load_raw_documents(tmp_path)

    assert acme.metadata == {
        "source": "acme.md",
        "entity": "Acme",
        "doc_type": "competitor_profile",
        "as_of": "2026-03",
    }
    assert acme.page_content == "# Acme\n\nText."
    assert plain.metadata == {"source": "plain.md"}


def test_a_leading_horizontal_rule_is_not_a_header() -> None:
    text = "---\n# Title\nIntro para.\n\n---\n\n## Section\ntext\n"

    assert parse_front_matter(text) == ({}, text)
