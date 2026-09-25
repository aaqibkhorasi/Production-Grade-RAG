"""Offline quality evaluation against the golden Q&A set.

Runs every golden question through the real query graph in-process (no HTTP
call -- this is testing pipeline quality, not the API's serialization layer).

Hard gates (fail CI): "answerable" cases must be grounded and must cite the
expected source document -- both come straight from our own deterministic
pipeline state, not an LLM judge, so they're reliable enough to block on.
"should_decline" cases must have grounded=False.

Tracked signals (recorded, never fail CI): Faithfulness, ContextualPrecision
and ContextualRelevancy, scored by a DeepEval/Bedrock judge. Each metric is
measured directly rather than through assert_test, so a score is a value this
suite owns and reports (see conftest.py) instead of warning text buried in the
log. Nothing about them can fail the run, which also removes the need for
DeepEval's `flaky` handling -- that only downgrades assert_test's own
AssertionError and never covered a crash inside metric computation.

All three were tried as hard gates first and dropped to non-blocking after
live investigation showed their failures traced to judge limitations, not
pipeline defects, on this corpus:
- Faithfulness: manually reproducing generate()+grounding_check() 4x on a
  failing case gave the identical, correct answer and GROUNDED=True every
  time -- the pipeline is deterministic and correct. DeepEval's own
  truths-decomposition step, inspected via verbose_mode, silently dropped an
  exception clause that was verbatim present in the context, then flagged an
  accurate claim as unsupported. That extraction is itself LLM-sampled and
  swings between runs on long, multi-chunk contexts.
- ContextualPrecision / ContextualRelevancy: Bedrock's Cohere reranker
  returns near-uniform scores (within ~0.04) across chunks from different
  sections of the same source document on this corpus, including sections
  that don't address the specific question. Provision-aware chunking cleared
  ContextualPrecision entirely but left ContextualRelevancy flat, since that
  measures the proportion of retrieved context that is relevant and retrieval
  still returns a fixed TOP_K however many chunks help. Separately, Haiku as
  judge occasionally returned
  malformed structured output for these metrics' verdict schemas
  (DeepEvalError: invalid JSON; a Pydantic literal_error on a "partial"
  verdict outside the yes/no enum) -- a judge-model reliability limit, not a
  pipeline defect.

DeepEval's NonAdviceMetric was tried and dropped entirely (not made flaky):
it flagged accurate restatements of SOP/CFR policy -- e.g. "the Lender may
reduce the equity injection if it determines the Borrower has sufficient
liquidity" -- as financial/legal advice violations, scoring 0.0. It's tuned
for chatbots giving personalized advice, not for explaining what a
regulation permits; not salvageable by threshold tuning for this domain.

This suite calls real Bedrock models (generation + judge) and is not part of
the default `pytest` run (see pyproject.toml's testpaths). Run explicitly:

    pytest eval/
"""

import json
import os
from pathlib import Path

import pytest
from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.models import AmazonBedrockModel
from deepeval.test_case import LLMTestCase

from pipeline.graph import run_query

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_qa.json"
QUALITY_THRESHOLD = 0.7

GOLDEN_SET = json.loads(GOLDEN_SET_PATH.read_text())
ANSWERABLE_CASES = [c for c in GOLDEN_SET if c["category"] == "answerable"]
DECLINE_CASES = [c for c in GOLDEN_SET if c["category"] == "should_decline"]


@pytest.fixture(scope="module")
def judge_model():
    return AmazonBedrockModel(
        model=os.getenv("CHAT_MODEL_ID") or "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region=os.getenv("AWS_REGION") or "us-east-1",
    )


def _tracked_metrics(judge_model):
    return [
        FaithfulnessMetric(threshold=QUALITY_THRESHOLD, model=judge_model, penalize_ambiguous_claims=True),
        ContextualPrecisionMetric(threshold=QUALITY_THRESHOLD, model=judge_model),
        ContextualRelevancyMetric(threshold=QUALITY_THRESHOLD, model=judge_model),
    ]


@pytest.mark.parametrize("case", ANSWERABLE_CASES, ids=[c["id"] for c in ANSWERABLE_CASES])
def test_answerable_case_is_grounded_and_cites_expected_source(case, judge_model, metric_recorder):
    result = run_query(case["question"])
    citations = result.get("citations", [])
    cited_source_docs = [c["source_doc"] for c in citations]
    contexts = [c["chunk_text"] for c in citations]

    assert result["grounded"], f"{case['id']}: expected a grounded answer"
    assert case["expected_source_doc"] in cited_source_docs, (
        f"{case['id']}: expected a citation from {case['expected_source_doc']}, got {cited_source_docs}"
    )

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result["answer"],
        expected_output=case["reference_answer"],
        retrieval_context=contexts or [""],
    )
    for metric in _tracked_metrics(judge_model):
        try:
            metric.measure(test_case)
            metric_recorder.record(case["id"], metric.__name__, metric.score, QUALITY_THRESHOLD)
        except Exception as exc:
            # The judge's own tooling can crash before producing a score -- seen
            # in practice when Haiku's output contained an apostrophe that broke
            # DeepEval's regex-based JSON extraction. These are tracked signals,
            # so a judge-tooling crash is recorded and skipped, exactly as a low
            # score would be; it never fails the run.
            metric_recorder.record(case["id"], metric.__name__, None, QUALITY_THRESHOLD, error=repr(exc))


@pytest.mark.parametrize("case", DECLINE_CASES, ids=[c["id"] for c in DECLINE_CASES])
def test_should_decline_case_is_not_grounded(case):
    result = run_query(case["question"])
    assert result["grounded"] is False, f"{case['id']}: expected a decline, but grounded={result['grounded']}"
