from unittest.mock import Mock, patch

from pipeline.retrieve import retrieve


def test_retrieve_unions_vector_and_bm25_candidates_then_reranks():
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [["v1", "v2", "shared"]]}
    fake_collection.get.return_value = {
        "ids": ["v1", "v2", "shared", "b1"],
        "documents": ["vector doc 1", "vector doc 2", "shared doc", "bm25 doc"],
        "metadatas": [
            {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"},
            {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"},
            {"source_doc": "b.html", "effective_date": "2026-02-01", "program": "affiliation"},
            {"source_doc": "b.html", "effective_date": "2026-02-01", "program": "affiliation"},
        ],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    def fake_rerank(question, candidates, top_k, client=None):
        # Simulate the reranker picking the "shared" and "b1" chunks as most relevant,
        # regardless of which retrieval method originally surfaced them. Both score
        # closely so RELEVANCE_RATIO keeps them -- this test is about the union, and
        # the floors have their own tests.
        by_id = {c["chunk_id"]: c for c in candidates}
        scored = [{**by_id["shared"], "score": 0.9}, {**by_id["b1"], "score": 0.85}]
        return scored[:top_k]

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=["shared", "b1"]),
        patch("pipeline.retrieve.rerank", side_effect=fake_rerank),
    ):
        result = retrieve({"question": "what is affiliation"}, collection=fake_collection)

    assert result["question"] == "what is affiliation"
    # "b1" reached the output despite only BM25 surfacing it, which is what the
    # union exists for. Both come from one source so cross-source filtering,
    # which has its own tests, does not interfere.
    assert [c["chunk_text"] for c in result["retrieved_chunks"]] == ["shared doc", "bm25 doc"]
    assert result["top_rerank_score"] == 0.9

    fetch_ids = fake_collection.get.call_args.kwargs["ids"]
    assert set(fetch_ids) == {"v1", "v2", "shared", "b1"}


def test_retrieve_drops_reranked_candidates_below_the_relevance_floor():
    # Regression test: reranking always fills up to TOP_K slots regardless of
    # whether that many candidates are actually relevant, padding citations
    # with topically-adjacent-but-irrelevant chunks.
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [["v1", "v2"]]}
    fake_collection.get.return_value = {
        "ids": ["v1", "v2"],
        "documents": ["relevant doc", "irrelevant doc"],
        "metadatas": [
            {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"},
            {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"},
        ],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    def fake_rerank(question, candidates, top_k, client=None):
        by_id = {c["chunk_id"]: c for c in candidates}
        return [{**by_id["v1"], "score": 0.8}, {**by_id["v2"], "score": 0.05}][:top_k]

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]),
        patch("pipeline.retrieve.rerank", side_effect=fake_rerank),
    ):
        result = retrieve({"question": "what is the equity requirement"}, collection=fake_collection)

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["chunk_text"] == "relevant doc"
    # The gate still checks against the single best score, even though the
    # low-scoring candidate was dropped from the citations shown to the user.
    assert result["top_rerank_score"] == 0.8


def test_retrieve_returns_empty_chunks_when_no_candidates_found():
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [[]]}
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]),
    ):
        result = retrieve({"question": "anything"}, collection=fake_collection)

    assert result["retrieved_chunks"] == []
    assert result["top_rerank_score"] == 0.0
    fake_collection.get.assert_not_called()


def test_retrieve_folds_recent_history_into_the_search_query():
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [[]]}
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    state = {
        "question": "Does that change if it's a start-up business?",
        "history": [
            {"role": "human", "content": "What is the minimum equity injection?"},
            {"role": "ai", "content": "The minimum equity injection is 10%."},
        ],
    }

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]) as mock_bm25,
    ):
        retrieve(state, collection=fake_collection)

    embedded_text = fake_embedder.embed_query.call_args[0][0]
    bm25_query = mock_bm25.call_args[0][0]
    assert "minimum equity injection" in embedded_text
    assert "start-up business" in embedded_text
    assert embedded_text == bm25_query


def test_retrieve_reranks_against_literal_question_not_folded_history():
    # Regression test: reranking against the history-folded search text let an
    # unrelated follow-up inherit a high relevance score from the *previous*
    # turn's topic, causing grounding_gate to incorrectly pass it through.
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [["v1"]]}
    fake_collection.get.return_value = {
        "ids": ["v1"],
        "documents": ["some doc about equity injection"],
        "metadatas": [{"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"}],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    state = {
        "question": "What is the capital of France?",
        "history": [
            {"role": "human", "content": "What is the minimum equity injection?"},
            {"role": "ai", "content": "The minimum equity injection is 10%."},
        ],
    }

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=["v1"]),
        patch("pipeline.retrieve.rerank", return_value=[]) as mock_rerank,
    ):
        retrieve(state, collection=fake_collection)

    rerank_question = mock_rerank.call_args[0][0]
    assert rerank_question == "What is the capital of France?"
    assert "equity injection" not in rerank_question


def test_retrieve_drops_candidates_far_below_the_best_score():
    # A chunk can clear the absolute floor and still be padding: here 0.40 is
    # well above it, but less than 85% of the 0.90 top score, so the reranker
    # is saying it is not in the same league as the best match.
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [["v1", "v2", "v3"]]}
    metadata = {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"}
    fake_collection.get.return_value = {
        "ids": ["v1", "v2", "v3"],
        "documents": ["best", "close second", "padding"],
        "metadatas": [metadata, metadata, metadata],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    def fake_rerank(question, candidates, top_k, client=None):
        by_id = {c["chunk_id"]: c for c in candidates}
        scores = {"v1": 0.90, "v2": 0.80, "v3": 0.40}
        return [{**by_id[i], "score": s} for i, s in scores.items()][:top_k]

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]),
        patch("pipeline.retrieve.rerank", side_effect=fake_rerank),
    ):
        result = retrieve({"question": "what is the equity requirement"}, collection=fake_collection)

    assert [c["chunk_text"] for c in result["retrieved_chunks"]] == ["best", "close second"]


def test_retrieve_still_applies_the_absolute_floor_when_every_candidate_is_weak():
    # Everything scores 100% of the top score, so the relative floor alone
    # would keep a whole set of irrelevant chunks for a question the corpus
    # cannot answer. The absolute floor is what stops that.
    fake_collection = Mock()
    fake_collection.query.return_value = {"ids": [["v1", "v2"]]}
    metadata = {"source_doc": "a.docx", "effective_date": "2026-01-01", "program": "core"}
    fake_collection.get.return_value = {
        "ids": ["v1", "v2"],
        "documents": ["weak", "also weak"],
        "metadatas": [metadata, metadata],
    }
    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]

    def fake_rerank(question, candidates, top_k, client=None):
        by_id = {c["chunk_id"]: c for c in candidates}
        return [{**by_id["v1"], "score": 0.05}, {**by_id["v2"], "score": 0.05}][:top_k]

    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]),
        patch("pipeline.retrieve.rerank", side_effect=fake_rerank),
    ):
        result = retrieve({"question": "what is the capital of France"}, collection=fake_collection)

    assert result["retrieved_chunks"] == []


def _collection_with(documents, sources):
    collection = Mock()
    ids = [f"c{i}" for i in range(len(documents))]
    collection.query.return_value = {"ids": [ids]}
    collection.get.return_value = {
        "ids": ids,
        "documents": documents,
        "metadatas": [
            {"source_doc": s, "effective_date": "2026-01-01", "program": "core"} for s in sources
        ],
    }
    return collection, ids


def _retrieve_with(collection, ids, scores, question="what is the upfront fee"):
    def fake_rerank(_question, candidates, top_k, client=None):
        by_id = {c["chunk_id"]: c for c in candidates}
        return [{**by_id[i], "score": s} for i, s in zip(ids, scores)][:top_k]

    fake_embedder = Mock()
    fake_embedder.embed_query.return_value = [0.1, 0.2]
    with (
        patch("pipeline.retrieve.get_embedding_model", return_value=fake_embedder),
        patch("pipeline.retrieve.rank_by_bm25", return_value=[]),
        patch("pipeline.retrieve.rerank", side_effect=fake_rerank),
    ):
        return retrieve({"question": question}, collection=collection)


def test_retrieve_drops_other_sources_that_merely_clear_the_relevance_floor():
    # Asked about a 7(a) fee, retrieval returned the right fee-notice chunk plus
    # SOP sections that also discuss fees -- close enough to survive the floor,
    # unrelated enough to dilute the context.
    collection, ids = _collection_with(
        ["fee tiers", "fee timing", "sop fees A", "sop fees B"],
        ["notice.pdf", "notice.pdf", "sop.docx", "sop.docx"],
    )

    result = _retrieve_with(collection, ids, [0.93, 0.90, 0.86, 0.80])

    assert [c["source_doc"] for c in result["retrieved_chunks"]] == ["notice.pdf", "notice.pdf"]


def test_retrieve_keeps_a_second_source_that_ties_with_the_best_chunk():
    # When the reranker cannot separate two documents, the question may genuinely
    # span both, so the tie window keeps them.
    collection, ids = _collection_with(
        ["cfr affiliation", "sop affiliation"], ["cfr.html", "sop.docx"]
    )

    result = _retrieve_with(collection, ids, [0.90, 0.895], question="what is affiliation")

    assert [c["source_doc"] for c in result["retrieved_chunks"]] == ["cfr.html", "sop.docx"]
