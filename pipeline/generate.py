import logging
import time
from pathlib import Path

import yaml

from config.settings import get_chat_model
from pipeline.state import Citation, QueryState

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"

logger = logging.getLogger(__name__)


def _load_prompt() -> dict:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["generate_answer"]


def _format_context(chunks: list[Citation]) -> str:
    return "\n\n".join(f"[{c['source_doc']}]\n{c['chunk_text']}" for c in chunks)


def _build_messages(question: str, chunks: list[Citation], history: list[dict] | None = None) -> list[tuple[str, str]]:
    prompt = _load_prompt()
    human = prompt["human_template"].format(question=question, context=_format_context(chunks))
    messages = [("system", prompt["system"])]
    for turn in history or []:
        messages.append((turn["role"], turn["content"]))
    messages.append(("human", human))
    return messages


def generate(state: QueryState, chat_model=None) -> QueryState:
    start = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])
    history = state.get("history") or []
    messages = _build_messages(state["question"], chunks, history)
    model = chat_model if chat_model is not None else get_chat_model()

    response = model.invoke(messages)

    duration_ms = (time.perf_counter() - start) * 1000
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    tokens_per_sec = None
    output_tokens = usage.get("output_tokens")
    if isinstance(output_tokens, (int, float)) and output_tokens and duration_ms > 0:
        tokens_per_sec = output_tokens / (duration_ms / 1000)
    logger.info(
        "generate: duration_ms=%.1f input_tokens=%s output_tokens=%s total_tokens=%s tokens_per_sec=%s",
        duration_ms,
        usage.get("input_tokens"),
        output_tokens,
        usage.get("total_tokens"),
        f"{tokens_per_sec:.1f}" if tokens_per_sec is not None else None,
    )

    return {**state, "answer": response.content, "citations": chunks}
