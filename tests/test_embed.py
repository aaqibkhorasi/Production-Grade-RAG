from unittest.mock import Mock, patch

import chromadb

from ingestion.chunker import Chunk
from ingestion.embed import upsert_chunks


def _make_chunk(chunk_id: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=f"chunk text {index}", source_doc="sop.docx",
        effective_date="2026-10-01", program="core", chunk_index=index,
    )


def test_upsert_chunks_writes_to_collection():
    collection = chromadb.EphemeralClient().get_or_create_collection("test-write")
    chunks = [_make_chunk("a1", 0), _make_chunk("a2", 1)]
    fake_embedder = Mock()
    fake_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with patch("ingestion.embed.get_embedding_model", return_value=fake_embedder):
        count = upsert_chunks(chunks, collection=collection)

    assert count == 2
    assert collection.count() == 2


def test_upsert_chunks_is_idempotent_on_rerun():
    collection = chromadb.EphemeralClient().get_or_create_collection("test-idempotent")
    chunks = [_make_chunk("b1", 0)]
    fake_embedder = Mock()
    fake_embedder.embed_documents.return_value = [[0.1, 0.2]]

    with patch("ingestion.embed.get_embedding_model", return_value=fake_embedder):
        upsert_chunks(chunks, collection=collection)
        upsert_chunks(chunks, collection=collection)

    assert collection.count() == 1


def test_upsert_chunks_returns_zero_for_empty_list():
    collection = chromadb.EphemeralClient().get_or_create_collection("test-empty")
    assert upsert_chunks([], collection=collection) == 0


def test_delete_chunks_for_source_removes_only_matching_chunks():
    collection = chromadb.EphemeralClient().get_or_create_collection("test-delete")
    old_chunk = _make_chunk("old1", 0)
    other_doc_chunk = Chunk(
        chunk_id="other1", text="unrelated", source_doc="other.docx",
        effective_date="2026-01-01", program="core", chunk_index=0,
    )
    fake_embedder = Mock()
    fake_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with patch("ingestion.embed.get_embedding_model", return_value=fake_embedder):
        upsert_chunks([old_chunk, other_doc_chunk], collection=collection)

    from ingestion.embed import delete_chunks_for_source
    delete_chunks_for_source("sop.docx", collection=collection)

    assert collection.count() == 1
    remaining = collection.get()
    assert remaining["metadatas"][0]["source_doc"] == "other.docx"
