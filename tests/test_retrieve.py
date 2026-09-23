from unittest.mock import Mock, patch

from pipeline.retrieve import retrieve


def test_retrieve_returns_chunks_from_collection():
    fake_collection = Mock()
    fake_collection.query.return_value = {
        "documents": [["chunk one text", "chunk two text"]],
        "metadatas": [
            [
                {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_index": 0},
                {"source_doc": "cfr.html", "effective_date": "2026-09-22", "program": "affiliation", "chunk_index": 3},
            ]
        ],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    with patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder):
        result = retrieve({"question": "What is affiliation?"}, collection=fake_collection)

    assert result["question"] == "What is affiliation?"
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["source_doc"] == "sop.docx"
    assert result["retrieved_chunks"][1]["chunk_text"] == "chunk two text"
