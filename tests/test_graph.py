from unittest.mock import patch


def test_run_query_wires_retrieve_and_generate():
    def fake_retrieve(state, collection=None):
        return {
            **state,
            "retrieved_chunks": [
                {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "context"}
            ],
        }

    def fake_generate(state, chat_model=None):
        return {**state, "answer": "final answer", "citations": state["retrieved_chunks"]}

    with (
        patch("pipeline.graph.retrieve", side_effect=fake_retrieve),
        patch("pipeline.graph.generate", side_effect=fake_generate),
    ):
        from pipeline.graph import run_query

        result = run_query("What is required?")

    assert result["answer"] == "final answer"
    assert result["citations"][0]["source_doc"] == "sop.docx"
