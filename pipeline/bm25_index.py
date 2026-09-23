import re

from rank_bm25 import BM25Okapi

_cache: dict = {"bm25": None, "ids": None}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(collection) -> None:
    """(Re)build the in-process BM25 index from every chunk currently in the collection."""
    result = collection.get(include=["documents"])
    ids = result["ids"]
    documents = result["documents"]
    tokenized_corpus = [_tokenize(doc) for doc in documents]
    _cache["bm25"] = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _cache["ids"] = ids


def invalidate_bm25_index() -> None:
    """Force the next call to rank_by_bm25 to rebuild from the collection's current contents.

    Call this after ingestion changes the corpus -- the index is cached in-process
    and does not otherwise notice new or changed chunks.
    """
    _cache["bm25"] = None
    _cache["ids"] = None


def rank_by_bm25(question: str, collection, top_k: int) -> list[str]:
    """Return up to top_k chunk ids ranked by BM25 keyword relevance to question."""
    if _cache["bm25"] is None:
        build_bm25_index(collection)
    bm25 = _cache["bm25"]
    ids = _cache["ids"]
    if not bm25 or not ids:
        return []

    scores = bm25.get_scores(_tokenize(question))
    ranked_indices = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [ids[i] for i in ranked_indices]
