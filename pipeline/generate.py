from pathlib import Path

import yaml

from config.settings import get_chat_model
from pipeline.state import Citation, QueryState

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


def _load_prompt() -> dict:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["generate_answer"]


def _format_context(chunks: list[Citation]) -> str:
    return "\n\n".join(f"[{c['source_doc']}]\n{c['chunk_text']}" for c in chunks)


def generate(state: QueryState, chat_model=None) -> QueryState:
    prompt = _load_prompt()
    chunks = state.get("retrieved_chunks", [])
    human = prompt["human_template"].format(question=state["question"], context=_format_context(chunks))
    model = chat_model if chat_model is not None else get_chat_model()

    response = model.invoke([("system", prompt["system"]), ("human", human)])

    return {**state, "answer": response.content, "citations": chunks}
