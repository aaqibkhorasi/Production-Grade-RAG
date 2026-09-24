from unittest.mock import Mock, patch

import docx

from ingestion.loader import load_document
from ingestion.manifest import ManifestEntry


def test_load_docx_extracts_paragraph_text(tmp_path):
    document = docx.Document()
    document.add_paragraph("SBA loans require a minimum equity injection.")
    document.add_paragraph("")
    document.add_paragraph("See Chapter 3 for eligibility.")
    document.save(str(tmp_path / "sop.docx"))
    entry = ManifestEntry(
        filename="sop.docx", url="https://example.com/sop.docx",
        file_type="docx", effective_date="2026-10-01", program="core",
    )

    result = load_document(entry, tmp_path)

    assert "minimum equity injection" in result.text
    assert "Chapter 3" in result.text
    assert result.source_doc == "sop.docx"
    assert result.effective_date == "2026-10-01"
    assert result.program == "core"


def test_load_pdf_extracts_page_text(tmp_path):
    (tmp_path / "notice.pdf").write_bytes(b"%PDF-fake")
    entry = ManifestEntry(
        filename="notice.pdf", url="https://example.com/notice.pdf",
        file_type="pdf", effective_date="2025-08-28", program="7(a)",
    )
    fake_page = Mock()
    fake_page.extract_text.return_value = "SBA guaranty fees for FY2026 are reduced."
    fake_reader = Mock(pages=[fake_page])

    with patch("ingestion.loader.PdfReader", return_value=fake_reader):
        result = load_document(entry, tmp_path)

    assert "guaranty fees" in result.text
    assert result.source_doc == "notice.pdf"


def test_load_html_extracts_visible_text(tmp_path):
    (tmp_path / "cfr.html").write_text(
        "<html><body><p>A business is affiliated with another if...</p></body></html>",
        encoding="utf-8",
    )
    entry = ManifestEntry(
        filename="cfr.html", url="https://example.com/cfr.html",
        file_type="html", effective_date="2026-09-22", program="affiliation",
    )

    result = load_document(entry, tmp_path)

    assert "affiliated with another" in result.text


def test_load_document_raises_on_empty_text(tmp_path):
    (tmp_path / "empty.html").write_text("<html><body></body></html>", encoding="utf-8")
    entry = ManifestEntry(
        filename="empty.html", url="https://example.com/empty.html",
        file_type="html", effective_date="2026-01-01", program="core",
    )

    try:
        load_document(entry, tmp_path)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_load_document_raises_when_extraction_falls_below_min_chars(tmp_path):
    # A bot-check page or JS shell parses fine and yields a little text, so the
    # empty-text check alone would let it through and silently poison retrieval.
    (tmp_path / "blocked.html").write_text(
        "<html><body><p>Please enable JavaScript to view this page.</p></body></html>",
        encoding="utf-8",
    )
    entry = ManifestEntry(
        filename="blocked.html", url="https://example.com/cfr.html",
        file_type="html", effective_date="2026-01-01", program="affiliation",
        min_chars=50_000,
    )

    try:
        load_document(entry, tmp_path)
        raised = False
    except ValueError as exc:
        raised = "expected at least 50000" in str(exc)
    assert raised


def test_load_html_prefers_the_main_content_container(tmp_path):
    # eCFR wraps the regulation in div.part; the surrounding page is nav and
    # browser-support banners that chunk into near-duplicate noise.
    (tmp_path / "cfr.html").write_text(
        "<html><body>"
        "<div class='banner'>You are using an unsupported browser</div>"
        "<div class='part'>stub</div>"
        "<div class='part'>Affiliation is based on control over the concern.</div>"
        "</body></html>",
        encoding="utf-8",
    )
    entry = ManifestEntry(
        filename="cfr.html", url="https://example.com/cfr.html",
        file_type="html", effective_date="2026-01-01", program="affiliation",
    )

    result = load_document(entry, tmp_path)

    assert "Affiliation is based on control" in result.text
    assert "unsupported browser" not in result.text
