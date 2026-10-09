"""Raw-document loaders for the ingestion pipeline."""

import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

_TEXT_SUFFIXES = {".md", ".txt"}
_PDF_SUFFIXES = {".pdf"}
_HEADER_KEYS = ("entity", "doc_type", "as_of")
_AS_OF = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_FENCE = "---\n"
_HEADER_KEY = re.compile(r"^[A-Za-z_]+$")


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading `---` block of `key: value` lines off a document.

    Returns (fields, body). Only `entity`, `doc_type` and `as_of` are kept,
    and `as_of` only in `YYYY-MM` form. No block, one that never closes, one
    with a line that isn't `key: value` (a document opening with a horizontal
    rule) or one naming none of those keys returns ({}, text) unchanged, so
    the document loads as before.
    """

    normalized = text.lstrip("\ufeff").replace("\r\n", "\n")
    if not normalized.startswith(_FENCE):
        return {}, text
    end = normalized.find("\n" + _FENCE, len(_FENCE) - 1)
    if end == -1:
        return {}, text
    fields: dict[str, str] = {}
    for line in normalized[len(_FENCE) : end].split("\n"):
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not _HEADER_KEY.match(key):
            # Not `key: value`: a leading horizontal rule, not a header.
            return {}, text
        if key in _HEADER_KEYS and value:
            fields[key] = value
    if not fields:
        return {}, text
    if "as_of" in fields and not _AS_OF.match(fields["as_of"]):
        del fields["as_of"]
    return fields, normalized[end + 1 + len(_FENCE) :]


def _load_text_file(path: Path, raw_dir: Path) -> Document:
    fields, body = parse_front_matter(path.read_text(encoding="utf-8"))
    return Document(
        page_content=body,
        metadata={"source": str(path.relative_to(raw_dir)), **fields},
    )


def _load_pdf_file(path: Path, raw_dir: Path) -> Document:
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(
        page_content=text,
        metadata={"source": str(path.relative_to(raw_dir)), "pages": len(reader.pages)},
    )


def load_raw_documents(raw_dir: Path) -> list[Document]:
    """Load every supported file under `raw_dir` into a `Document`.

    Supports plain text/markdown (`.md`, `.txt`) and `.pdf`. Unsupported
    file types are skipped rather than raising, so a data drop can mix in
    incidental files (e.g. `.DS_Store`) without breaking ingestion.
    """

    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            documents.append(_load_text_file(path, raw_dir))
        elif suffix in _PDF_SUFFIXES:
            documents.append(_load_pdf_file(path, raw_dir))
    return documents
