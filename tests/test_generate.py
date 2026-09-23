from unittest.mock import Mock

from pipeline.generate import generate


def test_generate_produces_answer_and_preserves_citations():
    retrieved_chunks = [
        {
            "source_doc": "sop.docx",
            "effective_date": "2026-10-01",
            "program": "core",
            "chunk_text": "Minimum equity injection is 10%.",
        }
    ]
    fake_model = Mock()
    fake_model.invoke.return_value = Mock(content="The minimum equity injection is 10%, per sop.docx.")

    result = generate(
        {"question": "What is the minimum equity injection?", "retrieved_chunks": retrieved_chunks},
        chat_model=fake_model,
    )

    assert result["answer"] == "The minimum equity injection is 10%, per sop.docx."
    assert result["citations"] == retrieved_chunks

    messages = fake_model.invoke.call_args[0][0]
    human_message = messages[1][1]
    assert "minimum equity injection" in human_message.lower()
    assert "sop.docx" in human_message
