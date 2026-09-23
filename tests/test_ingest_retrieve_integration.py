from unittest.mock import Mock, patch

import chromadb

from ingestion.chunker import Chunk
from ingestion.embed import upsert_chunks
from pipeline.bm25_index import invalidate_bm25_index
from pipeline.retrieve import retrieve


def _passthrough_rerank(question, candidates, top_k, client=None):
    return [{**c, "score": 1.0} for c in candidates[:top_k]]


def test_retrieve_round_trips_metadata_written_by_upsert_chunks():
    invalidate_bm25_index()
    collection = chromadb.EphemeralClient().get_or_create_collection("integration-test")
    chunks = [
        Chunk(
            chunk_id="i1",
            text="Minimum equity injection is 10%.",
            source_doc="sop.docx",
            effective_date="2026-10-01",
            program="core",
            chunk_index=0,
        )
    ]
    fake_embedder = Mock()
    fake_embedder.embed_documents.return_value = [[0.1, 0.2]]
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    with patch("ingestion.embed.get_embedding_model", return_value=fake_embedder):
        upsert_chunks(chunks, collection=collection)

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rerank", side_effect=_passthrough_rerank),
    ):
        result = retrieve({"question": "equity injection"}, collection=collection)

    assert len(result["retrieved_chunks"]) == 1
    citation = result["retrieved_chunks"][0]
    assert citation["source_doc"] == "sop.docx"
    assert citation["effective_date"] == "2026-10-01"
    assert citation["program"] == "core"
    assert citation["chunk_text"] == "Minimum equity injection is 10%."
