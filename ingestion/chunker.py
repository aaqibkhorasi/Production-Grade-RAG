import hashlib
from dataclasses import dataclass

import tiktoken

from ingestion.loader import Document

_ENCODING = tiktoken.get_encoding("cl100k_base")
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    effective_date: str
    program: str
    chunk_index: int


def chunk_document(document: Document) -> list[Chunk]:
    tokens = _ENCODING.encode(document.text)
    step = CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + CHUNK_MAX_TOKENS, len(tokens))
        chunk_text = _ENCODING.decode(tokens[start:end])
        chunk_id = hashlib.sha256(f"{document.source_doc}:{index}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_doc=document.source_doc,
                effective_date=document.effective_date,
                program=document.program,
                chunk_index=index,
            )
        )
        if end == len(tokens):
            break
        start += step
        index += 1

    return chunks
