from dataclasses import dataclass
from pathlib import Path

import docx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ingestion.manifest import ManifestEntry


@dataclass(frozen=True)
class Document:
    text: str
    source_doc: str
    effective_date: str
    program: str


def _load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return soup.get_text(separator="\n")


_LOADERS = {"docx": _load_docx, "pdf": _load_pdf, "html": _load_html}


def load_document(entry: ManifestEntry, raw_dir: Path) -> Document:
    path = raw_dir / entry.filename
    text = _LOADERS[entry.file_type](path)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}")
    return Document(
        text=text,
        source_doc=entry.filename,
        effective_date=entry.effective_date,
        program=entry.program,
    )
