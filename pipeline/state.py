from typing import TypedDict


class Citation(TypedDict):
    source_doc: str
    effective_date: str
    program: str
    chunk_text: str


class QueryState(TypedDict, total=False):
    question: str
    history: list[dict]
    retrieved_chunks: list[Citation]
    top_rerank_score: float
    gate_passed: bool
    answer: str
    citations: list[Citation]
    grounded: bool
