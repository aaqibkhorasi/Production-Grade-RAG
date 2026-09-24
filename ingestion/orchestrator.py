import logging
from pathlib import Path

from ingestion.chunker import chunk_document
from ingestion.embed import delete_chunks_for_source, upsert_chunks
from ingestion.loader import load_document
from ingestion.manifest import CORPUS_MANIFEST
from pipeline.bm25_index import invalidate_bm25_index

RAW_DIR = Path("data/raw")

logger = logging.getLogger(__name__)


def run_ingestion(raw_dir: Path = RAW_DIR) -> int:
    total = 0
    for entry in CORPUS_MANIFEST:
        document = load_document(entry, raw_dir)
        chunks = chunk_document(document)
        delete_chunks_for_source(entry.filename)
        total += upsert_chunks(chunks)
        # Per-document counts, not just the total: a source that silently
        # extracts or chunks to far less than expected still lets ingestion
        # "succeed" overall, and only shows up much later as that document
        # never being retrievable.
        logger.info(
            "ingested source_doc=%s text_chars=%d chunks=%d",
            entry.filename,
            len(document.text),
            len(chunks),
        )
    invalidate_bm25_index()
    logger.info("ingestion complete: sources=%d total_chunks=%d", len(CORPUS_MANIFEST), total)
    return total
