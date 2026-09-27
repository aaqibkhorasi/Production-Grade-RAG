# How this system is evaluated

Four numbers describe an answer's quality here. One of them fails the build; three are recorded and watched. This explains what each measures, how it is computed, and why they are treated differently.

The golden set is `eval/golden_qa.json`: 22 hand-written cases, 18 answerable and 4 that must be declined. Each answerable case carries the question, a reference answer, and the source document the answer should cite.

## `grounded` — the one that gates the build

This is not a DeepEval metric. It is a boolean the pipeline produces, and it is the only signal that can fail CI. Two independent checks must pass.

**Before generation — `grounding_gate`.** Did retrieval find anything relevant? The best rerank score is compared against `GROUNDING_GATE_THRESHOLD`. If nothing clears it the query is declined immediately, which also skips the cost of generation.

**After generation — `grounding_check`.** Two paths:

- *Deterministic.* If the generator emitted `INSUFFICIENT_CONTEXT`, decline. No model call.
- *Judged.* Otherwise an LLM is asked whether every factual claim in the draft follows from the retrieved chunks. This path is **fail-closed**: an exception, a timeout or an unparseable verdict all resolve to not-grounded, never to grounded.

`grounded=True` means the system retrieved something relevant, answered from it, and verified the answer against it. `grounded=False` means the user sees a decline rather than an answer.

The hard gates in CI are: an answerable case must be grounded **and** cite its expected source document; a decline case must not be grounded.

## Faithfulness — did the answer invent anything?

Takes the generated answer and the retrieved context. Three LLM passes:

1. **Extract truths from the context** — the facts inferable from the retrieved chunks.
2. **Extract claims from the answer** — the answer decomposed into atomic assertions.
3. **Judge each claim** — `yes` (agrees with the context), `no` (contradicts it), or `borderline`.

```
score = passing claims / total claims
```

We set `penalize_ambiguous_claims=True`, so only `yes` passes. The default also passes `borderline`, which means an unsupported-but-not-contradicting claim would count as faithful. In a regulatory domain that is the failure worth catching: "the documents do not say this, but it sounds right" is precisely what a lender must not act on.

**One detail matters more than the formula.** Claims are verified against the *extracted truths*, not the original chunks:

```python
retrieval_context="\n\n".join(self.truths)   # deepeval/metrics/faithfulness/faithfulness.py
```

Step 1 is lossy and sampled. A false hallucination flag in this project traced to exactly that: the truths extraction dropped an exception clause present verbatim in the retrieved chunk, and the judge then marked an accurate claim unsupported. This is the main reason the metric is tracked rather than gated.

## Contextual Precision — are the best chunks ranked first?

The only metric that needs the reference answer. For each retrieved chunk, in rank order, the judge is asked whether that chunk was useful in arriving at the expected answer. Then:

```
precision@k = (relevant chunks so far) / k        for each relevant chunk at rank k
score       = Σ precision@k / (total relevant chunks)
```

Five chunks, relevant at positions 1, 3 and 4:

| k | relevant | running count | precision@k |
| --- | --- | --- | --- |
| 1 | yes | 1 | 1/1 = 1.00 |
| 2 | no | — | — |
| 3 | yes | 2 | 2/3 = 0.67 |
| 4 | yes | 3 | 3/4 = 0.75 |
| 5 | no | — | — |

score = (1.00 + 0.67 + 0.75) / 3 = **0.81**

The same three chunks at positions 1, 2 and 3 score **1.00**. The metric is about ordering alone — it says nothing about how much irrelevant material came along.

## Contextual Relevancy — how much of the context was noise?

Each retrieved chunk is split into statements, and each statement is judged relevant to the **question** (not to the answer).

```
score = relevant statements / total statements
```

Being a ratio is what makes this metric behave unlike the others. Retrieving the perfect chunk alongside four neighbouring provisions still scores poorly, because the denominator grew. It is why better chunk boundaries left it unmoved while trimming the number of retrieved chunks improved it sharply: the fix has to shrink the denominator.

## What each one can and cannot see

| Metric | Catches | Blind to |
| --- | --- | --- |
| Faithfulness | The answer asserting things the sources do not support | Retrieving the wrong sources and summarising them accurately |
| Contextual Precision | Relevant chunks buried below irrelevant ones | Returning five chunks where one would do |
| Contextual Relevancy | Context padded with unrelated material | Whether the answer used any of it correctly |
| `grounded` | Answering a question the corpus cannot support | Anything about answer quality |

Faithfulness and Relevancy sit at opposite ends of the pipeline — one judges generation given the context, the other judges retrieval regardless of the answer. A case can score 1.00 on the first and 0.29 on the second.

## Why only one of them gates CI

`grounded` is computed from deterministic pipeline state: a score comparison and a string check. Re-running it gives the same answer.

The other three are LLM judgements, and their instability is measured rather than assumed. Across runs of identical code against an identical index, Faithfulness has come back with anywhere from 6 to 11 of 18 cases below threshold. A build that fails on that would fail at random, and a CI signal that fails at random is one people learn to ignore.

So they are recorded instead. Every run publishes a per-case table to the GitHub Actions job summary, with the count below threshold and the mean per metric. The **means** are the trustworthy part — they have moved consistently while the case counts swing.

A `NonAdvice` metric was tried and removed. It scored 0.0 on accurate restatements of SBA policy, which is what this system exists to produce; it is built for chatbots dispensing personal advice, not for explaining what a regulation permits.

## Running it

```bash
pytest eval/          # needs AWS credentials, ~8 minutes, real Bedrock spend
```

The table prints to the terminal afterwards. `EVAL_JUDGE_MODEL_ID` overrides the judge model if you want to trade cost against the occasional malformed verdict.

## What the measurements showed

Three retrieval changes were made and measured against the golden set. Recording which ones failed matters as much as which worked — the next person to look will otherwise repeat them.

| Change | Contextual Relevancy | Verdict |
| --- | --- | --- |
| Baseline: uniform 800-token chunks, fixed `TOP_K = 5` | 11 of 18 below threshold | — |
| Provision-aware chunking | 10 | **No effect on this metric** |
| Adaptive relevance floor (`RELEVANCE_RATIO`) | 3–7 | **Worked** — clears the noise band |
| Splitting wrapped PDF headings | unchanged | **No effect** |
| Cross-source filtering (`CROSS_SOURCE_TIE_RATIO`) | 4–6 | **Inside noise** — not proven |

**Chunking did not move Relevancy** because the metric is a ratio of relevant statements to total statements. Better chunk boundaries make each chunk more focused without changing that ratio while retrieval still returns a fixed five chunks. It did improve Faithfulness and Precision, and it fixed real defects, so it was kept.

**The relevance floor worked** because it attacks the denominator: it removes chunks that score far below the best one, so the padding disappears rather than being reorganised.

**Splitting wrapped PDF headings did not help,** though it was aimed squarely at the fee-notice cases. It is kept because it fixes a defect no judge is needed to see: the short-term fee rule had been filed under "for loans with a maturity that exceeds 12 months", so a citation shown to the user contradicted the text beneath it.

**Cross-source filtering** stops a fee question returning SOP sections alongside the fee notice. Measured across the golden set it never costs the expected source document and cuts average context from 3.6 chunks to 3.2, so fewer unrelated documents appear in the citation panel. Its effect on Relevancy is inside the spread, so it is not claimed as an improvement.

## Where the remaining failures come from

The cases still below the Relevancy threshold are mostly fee-notice questions, and the cause is now specific rather than general.

Asked *"what is the upfront fee for a short-term 7(a) loan with a maturity of 12 months or less?"*, retrieval returns two chunks, both from the right document, one of them exactly right. It still scores 0.25, because the reranker puts **"Upfront Fee for 7(a) WCP Loans"** first. Both chunks discuss 7(a) loans, maturity bands and a 0.25% rate, and the reranker cannot separate the standard programme from the Working Capital Pilot.

No filtering rule fixes this. Filtering removes chunks; it cannot reorder them, and here the problematic chunk is the top-ranked one. The remaining levers are a stronger reranker, query rewriting that names the programme explicitly, or simply more golden cases so that a change of this size is measurable at all.

That last point is the honest constraint on everything above: **22 cases judged by a sampled LLM cannot resolve differences of two or three cases.** Further tuning against this set would be fitting to noise.

## The judge was not running deterministically

Everything above about run-to-run spread was measured before noticing this: DeepEval builds its Bedrock request as

```python
"inferenceConfig": {**self.generation_kwargs}      # defaults to {}
```

With nothing passed, Bedrock applies the model's own default temperature and every verdict is a fresh sample. The pipeline being measured runs at `temperature=0`; the judge measuring it did not, so its sampling noise was being attributed to the code under test.

Pinning the judge to `temperature=0` changes the picture:

| | Run A | Run B |
| --- | --- | --- |
| Faithfulness | 9 below, mean 0.676 | 9 below, mean 0.676 |
| Contextual Precision | 0 below, mean 0.995 | 0 below, mean 0.995 |
| Contextual Relevancy | 4 below, mean 0.757 | 3 below, mean 0.759 |

Two consecutive runs, identical code and index: **17 of 18 cases scored identically**. The single case that moved, `sop-01`, went 0.70 to 0.73 and changed the count only because it sits exactly on the threshold.

So most of the instability documented earlier was configuration, not an inherent property of LLM-as-judge evaluation. Three caveats remain:

- **Determinism is not guaranteed even at temperature 0.** Greedy decoding still shifts with batching and hardware, and Bedrock makes no bitwise promise.
- **A threshold turns any residual wobble into a step change.** A case at 0.699 and one at 0.701 are the same answer and a different count. Means are steadier than counts for this reason.
- **Two runs is a small sample.** It is enough to show the spread collapsed, not enough to put a number on what remains.

The practical consequence is that changes previously dismissed as "inside the noise" — cross-source filtering in particular — are now worth re-measuring, because the instrument they were measured with was miscalibrated.
