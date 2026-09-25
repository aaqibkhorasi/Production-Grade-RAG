import logging
import time

from config.settings import get_embedding_model
from ingestion.embed import get_chroma_collection
from pipeline.bm25_index import rank_by_bm25
from pipeline.grounding_gate import GROUNDING_GATE_THRESHOLD
from pipeline.rerank import rerank
from pipeline.state import Citation, QueryState

TOP_K = 5  # ceiling, not a quota -- RELEVANCE_RATIO usually returns fewer
CANDIDATE_K = 15  # pool size per retrieval method, before reranking narrows to TOP_K

# Keep only chunks scoring within this fraction of the best chunk's score.
#
# An absolute floor cannot do this job: across the golden set the top score for
# a question the corpus cannot answer (0.72) sits above the second-best score
# for one it can (0.53), so any single cutoff either pads good answers or
# starves them. What does separate them is the shape of the curve -- a question
# with one right answer drops 40% after the top chunk, while a genuinely
# multi-chunk question stays flat -- so the floor is set relative to the top
# score and adapts per query.
#
# 0.85 was chosen by measuring every golden case: it takes the average answerable
# case from 5 chunks to 3.5 while still retrieving the expected source document
# for all 18, which is the citation hard gate.
RELEVANCE_RATIO = 0.85

logger = logging.getLogger(__name__)


def _build_retrieval_query(question: str, history: list[dict]) -> str:
    """Fold recent user turns into the retrieval query so pronoun-heavy
    follow-ups ("does that change if...") carry enough context to search well.
    Generation still sees the literal `question` plus the full history separately."""
    recent_user_turns = [turn["content"] for turn in history if turn.get("role") == "human"][-2:]
    return " ".join([*recent_user_turns, question])


def retrieve(state: QueryState, collection=None) -> QueryState:
    start = time.perf_counter()
    collection = collection if collection is not None else get_chroma_collection()
    embedder = get_embedding_model()
    question = state["question"]
    history = state.get("history") or []
    retrieval_query = _build_retrieval_query(question, history)

    query_embedding = embedder.embed_query(retrieval_query)
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=CANDIDATE_K)
    vector_ids = vector_results.get("ids", [[]])[0]

    bm25_ids = rank_by_bm25(retrieval_query, collection, top_k=CANDIDATE_K)

    candidate_ids = list(dict.fromkeys([*vector_ids, *bm25_ids]))  # union, de-duplicated, order preserved

    retrieved: list[Citation] = []
    top_rerank_score = 0.0
    if candidate_ids:
        fetched = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        candidates = [
            {"chunk_id": chunk_id, "text": text, "metadata": metadata}
            for chunk_id, text, metadata in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
        ]
        # Rerank against the literal question, not the history-folded retrieval_query --
        # scoring against the folded text let follow-ups unrelated to the current
        # question inherit a high score from the *previous* turn's relevance,
        # causing grounding_gate to pass questions it should have declined.
        reranked = rerank(question, candidates, top_k=TOP_K)
        if reranked:
            top_rerank_score = reranked[0]["score"]
        # Reranking alone always fills up to TOP_K slots regardless of whether
        # that many candidates are actually relevant -- padding the answer with
        # topically-adjacent-but-irrelevant chunks (e.g. other sections of the
        # same source document) that dilute both the generated answer's context
        # and the citations shown to the user.
        #
        # Two floors apply. The relative one trims that padding per query. The
        # absolute one is the same bar grounding_gate uses, and still matters
        # when every candidate is weak: without it a query the corpus cannot
        # answer would keep its best chunks purely for being the best of a bad
        # set, since they always score 100% of the top score.
        relevance_floor = max(GROUNDING_GATE_THRESHOLD, top_rerank_score * RELEVANCE_RATIO)
        top_candidates = [c for c in reranked if c["score"] >= relevance_floor]
        retrieved = [
            Citation(
                source_doc=c["metadata"]["source_doc"],
                effective_date=c["metadata"]["effective_date"],
                program=c["metadata"]["program"],
                chunk_text=c["text"],
            )
            for c in top_candidates
        ]

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "retrieve: question=%r vector_candidates=%d bm25_candidates=%d union_candidates=%d "
        "chunks_retrieved=%d top_rerank_score=%.4f duration_ms=%.1f",
        question,
        len(vector_ids),
        len(bm25_ids),
        len(candidate_ids),
        len(retrieved),
        top_rerank_score,
        duration_ms,
    )
    return {**state, "retrieved_chunks": retrieved, "top_rerank_score": top_rerank_score}
