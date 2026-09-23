from typing import TypedDict


class Citation(TypedDict):
    source_doc: str
    effective_date: str
    program: str
    chunk_text: str


class QueryState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[Citation]
    answer: str
    citations: list[Citation]
