import logging
import os
import time

import boto3

RERANK_MODEL_ID = "cohere.rerank-v3-5:0"

logger = logging.getLogger(__name__)


def rerank(question: str, candidates: list[dict], top_k: int, client=None) -> list[dict]:
    """Re-score candidates against question using Bedrock's hosted Cohere reranker.

    Each candidate must be a dict with at least a "text" key. Returns the top_k
    highest-scoring candidates, most relevant first.
    """
    if not candidates:
        return []

    start = time.perf_counter()
    # boto3's own default region resolution only reads AWS_DEFAULT_REGION, not
    # AWS_REGION -- which this project standardizes on everywhere else (see
    # .env.example). Locally that gap is masked by ~/.aws/config having a
    # default region; a clean CI runner has neither, so pass it explicitly.
    client = client if client is not None else boto3.client("bedrock-agent-runtime", region_name=os.getenv("AWS_REGION"))
    model_arn = f"arn:aws:bedrock:{client.meta.region_name}::foundation-model/{RERANK_MODEL_ID}"

    response = client.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": question}}],
        sources=[
            {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": c["text"]}}}
            for c in candidates
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": model_arn},
                "numberOfResults": min(top_k, len(candidates)),
            },
        },
    )

    ranked = [
        {**candidates[result["index"]], "score": result["relevanceScore"]} for result in response["results"]
    ]

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "rerank: question=%r candidates=%d returned=%d duration_ms=%.1f",
        question,
        len(candidates),
        len(ranked),
        duration_ms,
    )
    return ranked
