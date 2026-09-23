import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _reassemble_token_text(sse_body: str) -> str:
    """Reconstruct the full answer text from a raw SSE response body's token events."""
    pieces = []
    event_name = None
    for line in sse_body.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:") and event_name == "token":
            pieces.append(json.loads(line[len("data:") :].strip())["text"])
    return "".join(pieces)


def test_query_endpoint_returns_answer_citations_and_grounded_flag():
    fake_result = {
        "answer": "The minimum equity injection is 10%.",
        "citations": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "..."}
        ],
        "grounded": True,
    }
    with patch("api.main.run_query", return_value=fake_result):
        response = client.post("/query", json={"question": "What is the minimum equity injection?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_result["answer"]
    assert body["citations"][0]["source_doc"] == "sop.docx"
    assert body["grounded"] is True


def test_query_endpoint_surfaces_declined_answer():
    fake_result = {"answer": "I don't have enough relevant information...", "citations": [], "grounded": False}
    with patch("api.main.run_query", return_value=fake_result):
        response = client.post("/query", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["grounded"] is False


def test_query_endpoint_rejects_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_stream_endpoint_replays_a_verified_answer():
    fake_result = {
        "answer": "The minimum equity injection is 10%.",
        "citations": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "..."}
        ],
        "grounded": True,
    }
    with patch("api.main.run_query", return_value=fake_result) as mock_run_query:
        response = client.post(
            "/query/stream", json={"question": "What is the minimum equity injection?", "history": []}
        )

    assert response.status_code == 200
    body = response.text
    assert "event: citations" in body
    assert '"source_doc": "sop.docx"' in body
    assert "event: token" in body
    assert "event: grounded" in body
    assert '{"grounded": true}' in body
    assert "event: done" in body
    assert _reassemble_token_text(body) == "The minimum equity injection is 10%."

    mock_run_query.assert_called_once_with("What is the minimum equity injection?", history=[])


def test_query_stream_endpoint_surfaces_a_declined_answer():
    fake_result = {"answer": "I don't have enough relevant information...", "citations": [], "grounded": False}
    with patch("api.main.run_query", return_value=fake_result):
        response = client.post("/query/stream", json={"question": "What is the capital of France?", "history": []})

    assert response.status_code == 200
    body = response.text
    assert '"citations": []' in body
    assert '{"grounded": false}' in body
    assert _reassemble_token_text(body) == fake_result["answer"]


def test_query_stream_endpoint_passes_history_through_to_run_query():
    fake_result = {"answer": "It does not change.", "citations": [], "grounded": True}
    history = [{"role": "human", "content": "prior question"}, {"role": "ai", "content": "prior answer"}]

    with patch("api.main.run_query", return_value=fake_result) as mock_run_query:
        client.post("/query/stream", json={"question": "a follow-up", "history": history})

    mock_run_query.assert_called_once_with("a follow-up", history=history)


def test_ingest_endpoint_returns_chunk_count():
    with patch("api.main.run_ingestion", return_value=42):
        response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks_ingested": 42}
