from pathlib import Path

from ingestion.chunker import chunk_document
from ingestion.embed import delete_chunks_for_source, upsert_chunks
from ingestion.loader import load_document
from ingestion.manifest import CORPUS_MANIFEST

RAW_DIR = Path("data/raw")


def run_ingestion(raw_dir: Path = RAW_DIR) -> int:
    total = 0
    for entry in CORPUS_MANIFEST:
        document = load_document(entry, raw_dir)
        chunks = chunk_document(document)
        delete_chunks_for_source(entry.filename)
        total += upsert_chunks(chunks)
    return total
