import tiktoken

from ingestion.chunker import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, chunk_document
from ingestion.loader import Document

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _make_document(num_words: int) -> Document:
    text = " ".join(f"word{i}" for i in range(num_words))
    return Document(text=text, source_doc="test.docx", effective_date="2026-01-01", program="core")


def test_chunk_document_respects_max_token_size():
    chunks = chunk_document(_make_document(2000))
    for chunk in chunks[:-1]:
        assert len(_ENCODING.encode(chunk.text)) <= CHUNK_MAX_TOKENS


def test_chunk_document_overlap_between_consecutive_chunks():
    chunks = chunk_document(_make_document(2000))
    assert len(chunks) > 1
    first_tokens = _ENCODING.encode(chunks[0].text)
    second_tokens = _ENCODING.encode(chunks[1].text)
    assert second_tokens[:CHUNK_OVERLAP_TOKENS] == first_tokens[-CHUNK_OVERLAP_TOKENS:]


def test_chunk_document_inherits_metadata():
    chunks = chunk_document(_make_document(100))
    assert all(c.source_doc == "test.docx" for c in chunks)
    assert all(c.effective_date == "2026-01-01" for c in chunks)
    assert all(c.program == "core" for c in chunks)


def test_chunk_document_ids_are_deterministic():
    document = _make_document(1600)
    ids_a = [c.chunk_id for c in chunk_document(document)]
    ids_b = [c.chunk_id for c in chunk_document(document)]
    assert ids_a == ids_b


def test_chunk_document_indexes_are_sequential():
    chunks = chunk_document(_make_document(1600))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
