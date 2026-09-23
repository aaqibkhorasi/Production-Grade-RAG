from unittest.mock import Mock

from pipeline.rerank import rerank


def _fake_client(region_name: str = "us-east-1") -> Mock:
    client = Mock()
    client.meta.region_name = region_name
    return client


def test_rerank_reorders_candidates_by_relevance_score():
    candidates = [
        {"chunk_id": "a", "text": "unrelated fee schedule content", "metadata": {}},
        {"chunk_id": "b", "text": "minimum equity injection is 10 percent", "metadata": {}},
    ]
    client = _fake_client()
    client.rerank.return_value = {
        "results": [
            {"index": 1, "relevanceScore": 0.9},
            {"index": 0, "relevanceScore": 0.1},
        ]
    }

    result = rerank("what is the minimum equity injection", candidates, top_k=2, client=client)

    assert [c["chunk_id"] for c in result] == ["b", "a"]

    call_kwargs = client.rerank.call_args.kwargs
    assert call_kwargs["queries"][0]["textQuery"]["text"] == "what is the minimum equity injection"
    assert len(call_kwargs["sources"]) == 2
    assert call_kwargs["sources"][0]["inlineDocumentSource"]["textDocument"]["text"] == "unrelated fee schedule content"
    reranking_config = call_kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert reranking_config["numberOfResults"] == 2
    assert reranking_config["modelConfiguration"]["modelArn"].endswith("cohere.rerank-v3-5:0")
    assert "us-east-1" in reranking_config["modelConfiguration"]["modelArn"]


def test_rerank_returns_empty_list_for_no_candidates_without_calling_bedrock():
    client = _fake_client()

    result = rerank("question", [], top_k=5, client=client)

    assert result == []
    client.rerank.assert_not_called()


def test_rerank_caps_number_of_results_at_candidate_count():
    candidates = [{"chunk_id": "a", "text": "text", "metadata": {}}]
    client = _fake_client()
    client.rerank.return_value = {"results": [{"index": 0, "relevanceScore": 0.5}]}

    rerank("question", candidates, top_k=5, client=client)

    reranking_config = client.rerank.call_args.kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert reranking_config["numberOfResults"] == 1
