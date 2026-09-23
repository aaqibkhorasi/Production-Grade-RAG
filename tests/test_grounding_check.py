from unittest.mock import Mock

from pipeline.grounding_check import grounding_check


def _state_with_answer(answer: str) -> dict:
    return {
        "question": "What is the minimum equity injection?",
        "retrieved_chunks": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "Minimum equity injection is 10%."}
        ],
        "answer": answer,
        "citations": [
            {"source_doc": "sop.docx", "effective_date": "2026-10-01", "program": "core", "chunk_text": "Minimum equity injection is 10%."}
        ],
    }


def test_grounding_check_keeps_answer_when_model_says_grounded():
    fake_model = Mock()
    fake_model.invoke.return_value = Mock(content="GROUNDED\nThe answer matches the context exactly.")

    result = grounding_check(_state_with_answer("The minimum equity injection is 10%."), chat_model=fake_model)

    assert result["grounded"] is True
    assert result["answer"] == "The minimum equity injection is 10%."
    assert len(result["citations"]) == 1


def test_grounding_check_declines_when_model_says_not_grounded():
    fake_model = Mock()
    fake_model.invoke.return_value = Mock(content="NOT_GROUNDED\nThe context never mentions this claim.")

    result = grounding_check(_state_with_answer("A confident but unsupported claim."), chat_model=fake_model)

    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["answer"] != "A confident but unsupported claim."


def test_grounding_check_fails_closed_on_unparseable_response():
    fake_model = Mock()
    fake_model.invoke.return_value = Mock(content="I'm not sure how to answer that.")

    result = grounding_check(_state_with_answer("Some answer."), chat_model=fake_model)

    assert result["grounded"] is False
    assert result["citations"] == []


def test_grounding_check_fails_closed_when_model_call_raises():
    fake_model = Mock()
    fake_model.invoke.side_effect = RuntimeError("Bedrock throttled")

    result = grounding_check(_state_with_answer("Some answer."), chat_model=fake_model)

    assert result["grounded"] is False
    assert result["citations"] == []


def test_grounding_check_prompt_includes_context_and_draft_answer():
    fake_model = Mock()
    fake_model.invoke.return_value = Mock(content="GROUNDED")

    grounding_check(_state_with_answer("The minimum equity injection is 10%."), chat_model=fake_model)

    messages = fake_model.invoke.call_args[0][0]
    human_message = messages[1][1]
    assert "10%" in human_message
    assert "sop.docx" in human_message
