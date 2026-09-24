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


# Content containers to prefer, most specific first. eCFR wraps the actual
# regulation in div.part; taking the whole page instead pulls in ~50k chars of
# nav, cookie banners and "unsupported browser" notices, which chunk into dozens
# of near-duplicate fragments that dilute retrieval.
_HTML_CONTENT_SELECTORS = ("div.part", "main", "article", "#content")


def _load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for selector in _HTML_CONTENT_SELECTORS:
        matches = soup.select(selector)
        if matches:
            # A page can repeat a selector for small fragments (eCFR has a stub
            # div.part alongside the real one); the real content is the largest.
            return max(matches, key=lambda m: len(m.get_text())).get_text(separator="\n")
    return soup.get_text(separator="\n")


_LOADERS = {"docx": _load_docx, "pdf": _load_pdf, "html": _load_html}


def load_document(entry: ManifestEntry, raw_dir: Path) -> Document:
    path = raw_dir / entry.filename
    text = _LOADERS[entry.file_type](path)
    extracted = len(text.strip())
    if not extracted:
        raise ValueError(f"No text extracted from {path}")
    if extracted < entry.min_chars:
        raise ValueError(
            f"Only {extracted} chars extracted from {path} (expected at least "
            f"{entry.min_chars}) -- the download or parse likely failed"
        )
    return Document(
        text=text,
        source_doc=entry.filename,
        effective_date=entry.effective_date,
        program=entry.program,
    )
