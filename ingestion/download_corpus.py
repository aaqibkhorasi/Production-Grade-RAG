from pathlib import Path

import requests

from ingestion.manifest import CORPUS_MANIFEST, ManifestEntry

RAW_DIR = Path("data/raw")
USER_AGENT = "Mozilla/5.0 (compatible; AskMyDocsBot/1.0)"


def download_entry(entry: ManifestEntry, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / entry.filename
    if dest.exists():
        return dest
    response = requests.get(entry.url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def download_corpus(raw_dir: Path = RAW_DIR) -> list[Path]:
    return [download_entry(entry, raw_dir) for entry in CORPUS_MANIFEST]


if __name__ == "__main__":
    for path in download_corpus():
        print(f"downloaded: {path}")
