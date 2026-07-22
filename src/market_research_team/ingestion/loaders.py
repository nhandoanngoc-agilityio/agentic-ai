"""Raw-document loaders for the ingestion pipeline."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

_TEXT_SUFFIXES = {".md", ".txt"}
_PDF_SUFFIXES = {".pdf"}


def _load_text_file(path: Path, raw_dir: Path) -> Document:
    return Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path.relative_to(raw_dir))},
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
