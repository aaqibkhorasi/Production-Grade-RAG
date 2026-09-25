import tiktoken

from ingestion.chunker import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_STANDALONE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    chunk_document,
)
from ingestion.loader import Document, Section

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


def _sectioned(*sections) -> Document:
    return Document(
        text="\n".join(s.text for s in sections),
        source_doc="test.docx", effective_date="2026-01-01", program="core",
        sections=tuple(sections),
    )


def test_chunk_text_carries_the_heading_path():
    # Without this a chunk about EWCP fees never contains the word "EWCP", so
    # neither the embedding nor BM25 can match the question to it.
    document = _sectioned(
        Section(heading_path=("Section A", "Upfront Fee for EWCP loans"),
                text="The fee is 0.25% of the guaranteed portion."),
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert "Section A > Upfront Fee for EWCP loans" in chunks[0].text
    assert "0.25%" in chunks[0].text


def test_long_section_splits_on_line_boundaries_not_mid_provision():
    provisions = [f"({i}) Provision {i}: " + "detail " * 60 for i in range(20)]
    document = _sectioned(Section(heading_path=("Fees",), text="\n".join(provisions)))

    chunks = chunk_document(document)

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        for line in body.split("\n"):
            assert line in provisions, f"split landed mid-provision: {line[:60]!r}"


def test_small_sibling_sections_pack_into_one_chunk():
    # Otherwise each one-line section becomes its own tiny, low-signal chunk.
    document = _sectioned(
        Section(heading_path=("Questions",), text="Direct questions to the field office."),
        Section(heading_path=("Contact",), text="Call the Lender Relations Specialist."),
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert "field office" in chunks[0].text
    assert "Lender Relations" in chunks[0].text


def test_no_chunk_exceeds_the_hard_ceiling():
    document = _sectioned(
        Section(heading_path=("A",), text="\n".join("word " * 200 for _ in range(30))),
        Section(heading_path=("B",), text="single unbroken run " * 900),
    )

    for chunk in chunk_document(document):
        assert len(_ENCODING.encode(chunk.text)) <= CHUNK_MAX_TOKENS + 64  # + heading prefix


def test_sections_large_enough_to_stand_alone_are_not_packed_together():
    # Each of these is a distinct fee provision. Merging them is what let a
    # question about the annual service fee retrieve the upfront fee tiers too.
    body = "detail " * 150  # comfortably over CHUNK_MIN_STANDALONE_TOKENS (120)
    document = _sectioned(
        Section(heading_path=("Annual Service Fee",), text=body),
        Section(heading_path=("Upfront Fee for EWCP loans",), text=body),
    )

    assert len(_ENCODING.encode(body)) >= CHUNK_MIN_STANDALONE_TOKENS

    chunks = chunk_document(document)

    assert len(chunks) == 2
    assert "Annual Service Fee" in chunks[0].text
    assert "EWCP" not in chunks[0].text
