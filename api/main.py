import json
import logging
import re

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from api.schemas import CitationResponse, IngestResponse, QueryRequest, QueryResponse
from ingestion.orchestrator import run_ingestion
from pipeline.graph import run_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Ask My Docs - SBA RAG")


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _split_for_replay(text: str) -> list[str]:
    """Split a fully-generated, already-verified answer into small pieces so
    the client can render it progressively, without re-generating anything."""
    return re.findall(r"\S+\s*", text)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    history = [turn.model_dump() for turn in request.history]
    result = run_query(request.question, history=history)
    return QueryResponse(
        answer=result["answer"],
        citations=[CitationResponse(**c) for c in result["citations"]],
        grounded=result["grounded"],
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    def event_source():
        history = [turn.model_dump() for turn in request.history]
        # Correctness-first: run the full graph (retrieve -> grounding_gate ->
        # generate -> grounding_check) before showing anything to the client.
        # The answer is only ever "streamed" after it has already been verified --
        # this endpoint never shows text the grounding check hasn't approved.
        result = run_query(request.question, history=history)

        yield _sse("citations", {"citations": result["citations"]})

        for piece in _split_for_replay(result["answer"]):
            yield _sse("token", {"text": piece})

        yield _sse("grounded", {"grounded": result["grounded"]})
        yield _sse("done", {})

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    return IngestResponse(chunks_ingested=run_ingestion())
