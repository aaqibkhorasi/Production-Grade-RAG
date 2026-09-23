from unittest.mock import Mock, patch

from ingestion.download_corpus import download_corpus, download_entry
from ingestion.manifest import CORPUS_MANIFEST, ManifestEntry


def test_download_entry_writes_file(tmp_path):
    entry = ManifestEntry(
        filename="test.pdf", url="https://example.com/test.pdf",
        file_type="pdf", effective_date="2026-01-01", program="core",
    )
    fake_response = Mock(content=b"pdf-bytes")
    fake_response.raise_for_status = Mock()

    with patch("ingestion.download_corpus.requests.get", return_value=fake_response) as mock_get:
        dest = download_entry(entry, raw_dir=tmp_path)

    assert dest == tmp_path / "test.pdf"
    assert dest.read_bytes() == b"pdf-bytes"
    mock_get.assert_called_once()


def test_download_entry_skips_existing_file(tmp_path):
    entry = ManifestEntry(
        filename="test.pdf", url="https://example.com/test.pdf",
        file_type="pdf", effective_date="2026-01-01", program="core",
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    existing = tmp_path / "test.pdf"
    existing.write_bytes(b"already-here")

    with patch("ingestion.download_corpus.requests.get") as mock_get:
        dest = download_entry(entry, raw_dir=tmp_path)

    mock_get.assert_not_called()
    assert dest.read_bytes() == b"already-here"


def test_download_corpus_downloads_all_manifest_entries(tmp_path):
    fake_response = Mock(content=b"bytes")
    fake_response.raise_for_status = Mock()

    with patch("ingestion.download_corpus.requests.get", return_value=fake_response):
        paths = download_corpus(raw_dir=tmp_path)

    assert len(paths) == len(CORPUS_MANIFEST)
