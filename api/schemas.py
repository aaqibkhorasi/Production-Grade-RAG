from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    source_doc: str
    effective_date: str
    program: str
    chunk_text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IngestResponse(BaseModel):
    chunks_ingested: int
