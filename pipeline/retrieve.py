import logging
import time

from config.settings import get_embedding_model
from ingestion.embed import get_chroma_collection
from pipeline.bm25_index import rank_by_bm25
from pipeline.rerank import rerank
from pipeline.state import Citation, QueryState

TOP_K = 5
CANDIDATE_K = 15  # pool size per retrieval method, before reranking narrows to TOP_K

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
        top_candidates = rerank(question, candidates, top_k=TOP_K)
        if top_candidates:
            top_rerank_score = top_candidates[0]["score"]
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
