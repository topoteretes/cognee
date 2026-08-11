import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from cognee.modules.retrieval.mmr_retriever import MMRRetriever, _cosine_similarity, mmr_select


# --- pure-function tests ---

def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mmr_select_returns_top_k():
    q = [1.0, 0.0]
    # Three candidates; all slightly different
    cands = [[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]]
    result = mmr_select(q, cands, top_k=2, lambda_mult=0.5)
    assert len(result) == 2
    assert len(set(result)) == 2  # no duplicates


def test_mmr_select_empty_candidates():
    assert mmr_select([1.0, 0.0], [], top_k=3, lambda_mult=0.5) == []


def test_mmr_select_top_k_clamped_to_available():
    q = [1.0, 0.0]
    cands = [[0.9, 0.1], [0.8, 0.2]]
    result = mmr_select(q, cands, top_k=10, lambda_mult=0.5)
    assert len(result) == 2


def test_mmr_select_lambda_1_picks_by_relevance():
    # lambda=1 => pure relevance, no diversity penalty
    q = [1.0, 0.0]
    # candidate 0 is most relevant
    cands = [[0.99, 0.01], [0.5, 0.5], [0.1, 0.9]]
    result = mmr_select(q, cands, top_k=1, lambda_mult=1.0)
    assert result == [0]


def test_mmr_select_lambda_0_avoids_redundancy():
    # lambda=0 => pure diversity; second pick should differ from first
    q = [1.0, 0.0]
    cands = [[0.99, 0.01], [0.98, 0.02], [0.0, 1.0]]
    result = mmr_select(q, cands, top_k=2, lambda_mult=0.0)
    # First pick is maximally relevant (index 0 or 1), second must be diverse (index 2)
    assert 2 in result


# --- MMRRetriever tests ---

def _make_chunk(text: str):
    m = MagicMock()
    m.payload = {"text": text, "id": str(uuid4())}
    return m


def _patch_unified(candidates, embeddings):
    vector_engine = AsyncMock()
    vector_engine.search = AsyncMock(return_value=candidates)
    vector_engine.embedding_engine = AsyncMock()
    vector_engine.embedding_engine.embed_text = AsyncMock(return_value=embeddings)
    unified = AsyncMock()
    unified.vector = vector_engine
    return patch(
        "cognee.modules.retrieval.mmr_retriever.get_unified_engine",
        AsyncMock(return_value=unified),
    )


@pytest.mark.asyncio
async def test_get_retrieved_objects_returns_top_k():
    chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
    # query emb + 5 candidate embs; all distinct enough
    embs = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.85, 0.15],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.05, 0.95],
    ]
    with _patch_unified(chunks, embs):
        retriever = MMRRetriever(top_k=3, fetch_k=5)
        result = await retriever.get_retrieved_objects("test query")
    assert len(result) == 3


@pytest.mark.asyncio
async def test_get_retrieved_objects_empty_store():
    from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError

    vector_engine = AsyncMock()
    vector_engine.search = AsyncMock(side_effect=CollectionNotFoundError("collection missing"))
    unified = AsyncMock()
    unified.vector = vector_engine

    with patch(
        "cognee.modules.retrieval.mmr_retriever.get_unified_engine",
        AsyncMock(return_value=unified),
    ):
        retriever = MMRRetriever()
        with pytest.raises(Exception):
            await retriever.get_retrieved_objects("anything")


@pytest.mark.asyncio
async def test_get_context_from_objects_joins_text():
    chunks = [_make_chunk("alpha"), _make_chunk("beta")]
    retriever = MMRRetriever()
    ctx = await retriever.get_context_from_objects("q", chunks)
    assert ctx == "alpha\nbeta"


@pytest.mark.asyncio
async def test_get_context_from_objects_empty():
    retriever = MMRRetriever()
    ctx = await retriever.get_context_from_objects("q", [])
    assert ctx == ""


@pytest.mark.asyncio
async def test_get_completion_from_context_returns_payloads():
    chunks = [_make_chunk("x"), _make_chunk("y")]
    retriever = MMRRetriever()
    result = await retriever.get_completion_from_context("q", chunks, "x\ny")
    assert result == [c.payload for c in chunks]


@pytest.mark.asyncio
async def test_get_completion_from_context_empty():
    retriever = MMRRetriever()
    result = await retriever.get_completion_from_context("q", [], "")
    assert result == []


@pytest.mark.asyncio
async def test_no_candidates_returns_empty():
    with _patch_unified([], []):
        retriever = MMRRetriever()
        result = await retriever.get_retrieved_objects("q")
    assert result == []
