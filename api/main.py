from fastapi import FastAPI

from api.schemas import CitationResponse, IngestResponse, QueryRequest, QueryResponse
from ingestion.orchestrator import run_ingestion
from pipeline.graph import run_query

app = FastAPI(title="Ask My Docs - SBA RAG")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = run_query(request.question)
    return QueryResponse(
        answer=result["answer"],
        citations=[CitationResponse(**c) for c in result["citations"]],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    return IngestResponse(chunks_ingested=run_ingestion())
