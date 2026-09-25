import logging
from pathlib import Path

import yaml

from config.settings import get_chat_model
from pipeline.generate import _format_context
from pipeline.state import QueryState

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"

logger = logging.getLogger(__name__)


def _load_prompt() -> dict:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["grounding_check"]


def _decline_message() -> str:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["decline_message"].strip()


def _insufficient_context_marker() -> str:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["insufficient_context_marker"].strip()


def grounding_check(state: QueryState, chat_model=None) -> QueryState:
    """Post-generation verification: does the draft answer's content actually
    follow from the retrieved chunks? Fail-closed -- any error, exception, or
    unparseable model response defaults to NOT grounded, never to grounded."""
    prompt = _load_prompt()
    chunks = state.get("retrieved_chunks", [])

    # Deterministic decline, checked before the judge runs.
    #
    # `grounded` is a hard gate, and an LLM judge is the wrong thing to hang one
    # on. Observed directly: asked for the 504 interest rate cap, the generator
    # said the documents do not specify one and described 504 debenture pricing
    # instead. The judge called that GROUNDED on one context and NOT_GROUNDED on
    # another, because most of the draft's sentences were individually supported
    # -- the question going unanswered is not a property of any single sentence.
    # Having the generator emit a marker instead makes the decline a fact about
    # the pipeline rather than a judgement, and skips a model call.
    if _insufficient_context_marker() in (state.get("answer") or ""):
        logger.info("grounding_check: generator reported insufficient context, declining")
        return {**state, "answer": _decline_message(), "citations": [], "grounded": False}
    context = _format_context(chunks)
    human = prompt["human_template"].format(answer=state.get("answer", ""), context=context)
    model = chat_model if chat_model is not None else get_chat_model()

    grounded = False
    try:
        response = model.invoke([("system", prompt["system"]), ("human", human)])
        verdict = (response.content or "").strip().upper()
        grounded = verdict.startswith("GROUNDED")
    except Exception:
        logger.exception("grounding_check: model call failed, defaulting to not grounded (fail-closed)")

    if grounded:
        logger.info("grounding_check: answer is grounded")
        return {**state, "grounded": True, "citations": chunks}

    logger.info("grounding_check: answer is NOT grounded, declining")
    return {**state, "answer": _decline_message(), "citations": [], "grounded": False}
