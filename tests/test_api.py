from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_query_endpoint_returns_answer_and_citations():
    fake_result = {
        "answer": "The minimum equity injection is 10%.",
        "citations": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "..."}
        ],
    }
    with patch("api.main.run_query", return_value=fake_result):
        response = client.post("/query", json={"question": "What is the minimum equity injection?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_result["answer"]
    assert body["citations"][0]["source_doc"] == "sop.docx"


def test_query_endpoint_rejects_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_ingest_endpoint_returns_chunk_count():
    with patch("api.main.run_ingestion", return_value=42):
        response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks_ingested": 42}
