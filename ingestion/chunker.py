import hashlib
from dataclasses import dataclass

import tiktoken

from ingestion.loader import Document, Section

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Hard ceiling. Nothing is ever emitted above this.
CHUNK_MAX_TOKENS = 800
# Preferred size. Packing stops here, so a typical chunk is well under the
# ceiling -- a chunk padded out to 800 tokens carries several neighbouring
# provisions, which is what dilutes retrieval precision.
CHUNK_TARGET_TOKENS = 450
# Only used for a single unbroken run of text with no structure to split on.
CHUNK_OVERLAP_TOKENS = 100
# A section this size or larger becomes its own chunk rather than being packed
# with its neighbours.
#
# Packing everything up to the target merged whole provisions together: the fee
# notice's 14 sections collapsed into 6 chunks, so a question about the annual
# service fee retrieved a chunk that also held the upfront fee tiers. Since the
# notice is the entire corpus for fee questions, retrieval then returned most of
# the document and no relevance floor could trim it -- every chunk scored alike.
# Packing now applies only to fragments too small to stand on their own.
CHUNK_MIN_STANDALONE_TOKENS = 120


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    effective_date: str
    program: str
    chunk_index: int


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_unbroken_text(text: str) -> list[str]:
    """Last resort for a run of text with no line breaks that still exceeds the
    ceiling. There is nothing structural to cut on, so fall back to blind token
    windows with overlap so a sentence spanning the cut is not lost."""
    tokens = _ENCODING.encode(text)
    step = CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS
    pieces = []
    for start in range(0, len(tokens), step):
        pieces.append(_ENCODING.decode(tokens[start : start + CHUNK_MAX_TOKENS]))
        if start + CHUNK_MAX_TOKENS >= len(tokens):
            break
    return pieces


def _split_section_body(body: str) -> list[str]:
    """Pack whole lines -- provisions, list items, fee tiers -- up to the target
    size. A line is never broken, so a split lands between provisions."""
    pieces: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in (line for line in body.split("\n") if line.strip()):
        length = _token_count(line)
        if length > CHUNK_MAX_TOKENS:
            if buffer:
                pieces.append("\n".join(buffer))
                buffer, size = [], 0
            pieces.extend(_split_unbroken_text(line))
            continue
        if buffer and size + length > CHUNK_TARGET_TOKENS:
            pieces.append("\n".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += length
    if buffer:
        pieces.append("\n".join(buffer))
    return pieces


def chunk_document(document: Document) -> list[Chunk]:
    # A source with no detected headings still chunks, as one unlabelled section.
    sections = document.sections or (Section(heading_path=(), text=document.text),)

    texts: list[str] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            texts.append("\n\n".join(buffer))
            buffer, size = [], 0

    for section in sections:
        # The heading path rides along in the chunk text itself rather than in
        # metadata, so it reaches the embedding, the BM25 index, the generator's
        # context and the citation shown to the user without touching any schema.
        # Without it a chunk about EWCP fees never says the word "EWCP".
        heading = " > ".join(section.heading_path)
        body_size = _token_count(section.text)
        if body_size > CHUNK_TARGET_TOKENS:
            flush()
            for piece in _split_section_body(section.text):
                texts.append(f"{heading}\n\n{piece}" if heading else piece)
            continue
        # A section that can stand on its own does, so one provision stays one
        # chunk. Only fragments below that size pack together, and never across
        # the target.
        if body_size >= CHUNK_MIN_STANDALONE_TOKENS:
            flush()
            texts.append(f"{heading}\n\n{section.text}" if heading else section.text)
            continue
        if buffer and size + body_size > CHUNK_TARGET_TOKENS:
            flush()
        buffer.append(f"{heading}\n\n{section.text}" if heading else section.text)
        size += body_size
    flush()

    return [
        Chunk(
            chunk_id=hashlib.sha256(f"{document.source_doc}:{index}".encode()).hexdigest()[:16],
            text=text,
            source_doc=document.source_doc,
            effective_date=document.effective_date,
            program=document.program,
            chunk_index=index,
        )
        for index, text in enumerate(texts)
    ]
