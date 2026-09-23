from pipeline.grounding_gate import GROUNDING_GATE_THRESHOLD, grounding_gate


def test_grounding_gate_passes_when_score_meets_threshold():
    state = {
        "question": "What is the minimum equity injection?",
        "retrieved_chunks": [{"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "..."}],
        "top_rerank_score": GROUNDING_GATE_THRESHOLD + 0.1,
    }

    result = grounding_gate(state)

    assert result["gate_passed"] is True
    # Passing the gate must not fabricate an answer -- generate hasn't run yet.
    assert "answer" not in result


def test_grounding_gate_declines_when_score_below_threshold():
    state = {
        "question": "What is the capital of France?",
        "retrieved_chunks": [{"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "irrelevant"}],
        "top_rerank_score": GROUNDING_GATE_THRESHOLD - 0.1,
    }

    result = grounding_gate(state)

    assert result["gate_passed"] is False
    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["answer"]  # a real decline message, not empty


def test_grounding_gate_declines_when_no_chunks_retrieved_even_with_high_score():
    state = {"question": "anything", "retrieved_chunks": [], "top_rerank_score": 0.99}

    result = grounding_gate(state)

    assert result["gate_passed"] is False
    assert result["grounded"] is False
