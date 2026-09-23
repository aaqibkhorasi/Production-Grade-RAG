from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class CitationResponse(BaseModel):
    source_doc: str
    effective_date: str
    program: str
    chunk_text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    grounded: bool


class IngestResponse(BaseModel):
    chunks_ingested: int
