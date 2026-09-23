from unittest.mock import patch

from ingestion.chunker import Chunk
from ingestion.loader import Document
from ingestion.manifest import CORPUS_MANIFEST
from ingestion.orchestrator import run_ingestion


def test_run_ingestion_processes_all_manifest_entries(tmp_path):
    fake_document = Document(text="irrelevant", source_doc="x", effective_date="2026-01-01", program="core")
    fake_chunks = [
        Chunk(chunk_id="c1", text="t", source_doc="x", effective_date="2026-01-01", program="core", chunk_index=0)
    ]

    with (
        patch("ingestion.orchestrator.load_document", return_value=fake_document) as mock_load,
        patch("ingestion.orchestrator.chunk_document", return_value=fake_chunks) as mock_chunk,
        patch("ingestion.orchestrator.delete_chunks_for_source", return_value=None) as mock_delete,
        patch("ingestion.orchestrator.upsert_chunks", return_value=len(fake_chunks)) as mock_upsert,
    ):
        total = run_ingestion(raw_dir=tmp_path)

    assert mock_load.call_count == len(CORPUS_MANIFEST)
    assert mock_chunk.call_count == len(CORPUS_MANIFEST)
    assert mock_delete.call_count == len(CORPUS_MANIFEST)
    assert mock_upsert.call_count == len(CORPUS_MANIFEST)
    assert total == len(fake_chunks) * len(CORPUS_MANIFEST)
