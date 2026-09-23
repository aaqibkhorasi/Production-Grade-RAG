from unittest.mock import patch


def _fake_retrieve(state, collection=None):
    return {
        **state,
        "retrieved_chunks": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "context"}
        ],
        "top_rerank_score": 0.85,
    }


def test_run_query_full_path_when_gate_passes_and_answer_is_grounded():
    def fake_gate(state):
        return {**state, "gate_passed": True}

    def fake_generate(state, chat_model=None):
        return {**state, "answer": "final answer", "citations": state["retrieved_chunks"]}

    def fake_grounding_check(state, chat_model=None):
        return {**state, "grounded": True, "citations": state["citations"]}

    with (
        patch("pipeline.graph.retrieve", side_effect=_fake_retrieve),
        patch("pipeline.graph.grounding_gate", side_effect=fake_gate),
        patch("pipeline.graph.generate", side_effect=fake_generate),
        patch("pipeline.graph.grounding_check", side_effect=fake_grounding_check),
    ):
        from pipeline.graph import run_query

        result = run_query("What is required?")

    assert result["answer"] == "final answer"
    assert result["grounded"] is True
    assert result["citations"][0]["source_doc"] == "sop.docx"


def test_run_query_declines_early_when_gate_fails_without_calling_generate():
    def fake_gate(state):
        return {**state, "gate_passed": False, "answer": "I cannot answer this.", "citations": [], "grounded": False}

    with (
        patch("pipeline.graph.retrieve", side_effect=_fake_retrieve),
        patch("pipeline.graph.grounding_gate", side_effect=fake_gate),
        patch("pipeline.graph.generate") as mock_generate,
        patch("pipeline.graph.grounding_check") as mock_grounding_check,
    ):
        from pipeline.graph import run_query

        result = run_query("What is the capital of France?")

    assert result["answer"] == "I cannot answer this."
    assert result["grounded"] is False
    assert result["citations"] == []
    mock_generate.assert_not_called()
    mock_grounding_check.assert_not_called()


def test_run_query_declines_when_grounding_check_rejects_the_draft_answer():
    def fake_gate(state):
        return {**state, "gate_passed": True}

    def fake_generate(state, chat_model=None):
        return {**state, "answer": "a confident but unsupported answer", "citations": state["retrieved_chunks"]}

    def fake_grounding_check(state, chat_model=None):
        return {**state, "answer": "I cannot verify this answer.", "citations": [], "grounded": False}

    with (
        patch("pipeline.graph.retrieve", side_effect=_fake_retrieve),
        patch("pipeline.graph.grounding_gate", side_effect=fake_gate),
        patch("pipeline.graph.generate", side_effect=fake_generate),
        patch("pipeline.graph.grounding_check", side_effect=fake_grounding_check),
    ):
        from pipeline.graph import run_query

        result = run_query("What is required?")

    assert result["answer"] == "I cannot verify this answer."
    assert result["grounded"] is False
    assert result["citations"] == []
