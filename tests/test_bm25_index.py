import chromadb

from pipeline.bm25_index import build_bm25_index, invalidate_bm25_index, rank_by_bm25


def _make_collection_with_docs(name: str, docs: dict[str, str]):
    collection = chromadb.EphemeralClient().get_or_create_collection(name)
    collection.upsert(
        ids=list(docs.keys()),
        embeddings=[[0.0] for _ in docs],  # embeddings unused by BM25, dummy values
        documents=list(docs.values()),
        metadatas=[{"placeholder": True} for _ in docs],
    )
    return collection


def test_rank_by_bm25_ranks_keyword_relevant_docs_higher():
    invalidate_bm25_index()
    collection = _make_collection_with_docs(
        "bm25-relevance",
        {
            "a": "the minimum equity injection is ten percent",
            "b": "guaranty fees are assessed annually on the outstanding balance",
            "c": "equity injection requirements for start-up businesses",
        },
    )

    ranked = rank_by_bm25("equity injection requirements", collection, top_k=3)

    assert ranked[0] in ("a", "c")
    assert set(ranked) == {"a", "b", "c"}


def test_rank_by_bm25_returns_empty_list_for_empty_collection():
    invalidate_bm25_index()
    collection = chromadb.EphemeralClient().get_or_create_collection("bm25-empty")

    assert rank_by_bm25("anything", collection, top_k=5) == []


def test_invalidate_bm25_index_forces_rebuild_on_next_call():
    # BM25's IDF formula gives a term appearing in exactly half a tiny corpus a
    # score of exactly zero (log(1) == 0), so this needs enough unrelated
    # documents that "zephyranthes" appearing in only one of them is
    # meaningfully rare, not exactly half the corpus.
    invalidate_bm25_index()
    collection = _make_collection_with_docs(
        "bm25-invalidate",
        {
            "a": "equity injection ten percent",
            "c": "guaranty fees are assessed annually",
            "d": "affiliation based on identity of interest",
        },
    )
    build_bm25_index(collection)
    ranked_before = rank_by_bm25("zephyranthes", collection, top_k=1)
    assert ranked_before != ["b"]  # "b" doesn't exist in the index yet

    collection.upsert(
        ids=["b"],
        embeddings=[[0.0]],
        documents=["zephyranthes is a genus of flowering plants"],
        metadatas=[{"placeholder": True}],
    )
    invalidate_bm25_index()

    # After invalidation, the rebuilt index sees the new document and its unique term.
    assert rank_by_bm25("zephyranthes", collection, top_k=1) == ["b"]
