import logging
from pathlib import Path

import yaml

from pipeline.state import QueryState

# Not empirically calibrated -- a conservative starting point based on observed
# Bedrock Cohere rerank scores (clearly irrelevant candidates scored ~0.1,
# genuinely relevant ones scored 0.7-0.9). Revisit once real usage data exists.
GROUNDING_GATE_THRESHOLD = 0.2

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"

logger = logging.getLogger(__name__)


def _decline_message() -> str:
    return yaml.safe_load(_PROMPTS_PATH.read_text())["decline_message"].strip()


def grounding_gate(state: QueryState) -> QueryState:
    """Cheap pre-generation check: decline immediately if nothing relevant was
    retrieved, skipping the (expensive) generate and grounding_check steps entirely."""
    chunks = state.get("retrieved_chunks", [])
    top_score = state.get("top_rerank_score", 0.0)
    passed = bool(chunks) and top_score >= GROUNDING_GATE_THRESHOLD

    if passed:
        logger.info("grounding_gate: passed (top_rerank_score=%.4f)", top_score)
        return {**state, "gate_passed": True}

    logger.info(
        "grounding_gate: declining (chunks_retrieved=%d, top_rerank_score=%.4f, threshold=%.2f)",
        len(chunks),
        top_score,
        GROUNDING_GATE_THRESHOLD,
    )
    return {
        **state,
        "gate_passed": False,
        "answer": _decline_message(),
        "citations": [],
        "grounded": False,
    }
