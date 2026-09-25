import logging
import time
from pathlib import Path

import requests

from ingestion.manifest import CORPUS_MANIFEST, ManifestEntry

RAW_DIR = Path("data/raw")
USER_AGENT = "Mozilla/5.0 (compatible; AskMyDocsBot/1.0)"
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5

logger = logging.getLogger(__name__)


def download_entry(entry: ManifestEntry, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / entry.filename
    if dest.exists():
        return dest

    failure: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            response = requests.get(entry.url, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            failure = exc
            logger.warning("download %s attempt %d failed: %r", entry.filename, attempt + 1, exc)
            continue

        # A rate-limited or bot-challenged request answers 200 with a short page
        # rather than an error status, so raise_for_status sees nothing wrong.
        # eCFR did exactly this to a CI runner, and the corpus was rebuilt from
        # a 1.4k challenge page. Every real source is far larger than its
        # min_chars floor, so a body under that is not the document.
        if len(response.content) < entry.min_chars:
            failure = ValueError(
                f"{entry.url} returned {len(response.content)} bytes, below the "
                f"{entry.min_chars} expected -- likely rate limited or blocked"
            )
            logger.warning("download %s attempt %d: %s", entry.filename, attempt + 1, failure)
            continue

        dest.write_bytes(response.content)
        return dest

    raise RuntimeError(
        f"Could not download {entry.filename} after {DOWNLOAD_ATTEMPTS} attempts: {failure}"
    )


def download_corpus(raw_dir: Path = RAW_DIR) -> list[Path]:
    return [download_entry(entry, raw_dir) for entry in CORPUS_MANIFEST]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    for path in download_corpus():
        print(f"downloaded: {path}")
