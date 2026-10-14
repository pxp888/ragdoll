import pytest
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.clients.standard_reranker import StandardReranker


@pytest.mark.asyncio
async def test_live_embedding_client():
    client = EmbeddingClient()
    health = await client.check_health()
    assert health is True

    # Test single and batch embed
    texts = [
        "First test sentence for ragdoll embedding.",
        "Second test sentence covering multi-scale retrieval.",
    ]
    embeddings = await client.embed_batch(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768


@pytest.mark.asyncio
async def test_live_kev_client():
    client = KevClient()
    health = await client.check_health()
    assert health is True

    # Test boundary decision
    block_a = "Machine learning models rely on dense numerical representations called embeddings."
    block_b = "Tomorrow's weather forecast predicts a high of 22 degrees and light rain."
    is_split = await client.decide_boundary(block_a, block_b)
    assert isinstance(is_split, bool)

    # Test multi-candidate breakpoint decision
    candidates = [
        "Embeddings allow semantic search across large corpuses.",
        "Tomorrow's weather forecast predicts a high of 22 degrees and light rain.",
    ]
    breakpoint_idx = await client.decide_breakpoint(block_a, candidates)
    assert isinstance(breakpoint_idx, int)
    assert 0 <= breakpoint_idx <= len(candidates)

    # Test relevance scoring
    query = "dense embeddings in machine learning"
    passage = "Vector embeddings capture the semantic meaning of sentences in a high dimensional space."
    score = await client.score_relevance(query, passage)
    assert isinstance(score, float)
    assert score >= 0.0


@pytest.mark.asyncio
async def test_standard_reranker():
    client = StandardReranker()
    health = await client.check_health()
    assert health is True

    passages = [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
        {"id": 2, "text": "Deep learning models require GPUs for accelerated matrix operations."},
    ]
    results = await client.rerank(query="GPU acceleration deep learning", passages=passages)
    assert len(results) == 2
    # Passage 2 should rank first
    assert results[0]["id"] == 2
    assert "score" in results[0]
    assert results[0]["score"] > results[1]["score"]
