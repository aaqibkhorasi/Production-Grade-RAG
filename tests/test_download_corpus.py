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
    # Body has to clear the largest min_chars in the manifest: a short response
    # is now treated as a block and retried rather than written.
    largest = max(entry.min_chars for entry in CORPUS_MANIFEST)
    fake_response = Mock(content=b"x" * (largest + 1))
    fake_response.raise_for_status = Mock()

    with patch("ingestion.download_corpus.requests.get", return_value=fake_response):
        paths = download_corpus(raw_dir=tmp_path)

    assert len(paths) == len(CORPUS_MANIFEST)


def test_download_entry_retries_when_the_body_is_too_small(tmp_path):
    # A rate-limited or bot-challenged request answers 200 with a short page, so
    # raise_for_status sees nothing wrong. eCFR did this to a CI runner and the
    # corpus was rebuilt from a 1.4k challenge page.
    entry = ManifestEntry(
        filename="cfr.html", url="https://example.com/cfr.html",
        file_type="html", effective_date="2026-01-01", program="affiliation",
        min_chars=50_000,
    )
    blocked = Mock(content=b"<html>Too many requests</html>")
    blocked.raise_for_status.return_value = None
    good = Mock(content=b"x" * 60_000)
    good.raise_for_status.return_value = None

    with (
        patch("ingestion.download_corpus.requests.get", side_effect=[blocked, good]) as get,
        patch("ingestion.download_corpus.time.sleep"),
    ):
        result = download_entry(entry, tmp_path)

    assert get.call_count == 2
    assert result.read_bytes() == b"x" * 60_000


def test_download_entry_raises_after_exhausting_attempts(tmp_path):
    entry = ManifestEntry(
        filename="cfr.html", url="https://example.com/cfr.html",
        file_type="html", effective_date="2026-01-01", program="affiliation",
        min_chars=50_000,
    )
    blocked = Mock(content=b"<html>Too many requests</html>")
    blocked.raise_for_status.return_value = None

    with (
        patch("ingestion.download_corpus.requests.get", return_value=blocked),
        patch("ingestion.download_corpus.time.sleep"),
    ):
        try:
            download_entry(entry, tmp_path)
            raised = False
        except RuntimeError as exc:
            raised = "likely rate limited or blocked" in str(exc)

    assert raised
    # A short page must never be written -- the loader guard would catch it, but
    # only after the cache key has already been poisoned for later runs.
    assert not (tmp_path / "cfr.html").exists()
