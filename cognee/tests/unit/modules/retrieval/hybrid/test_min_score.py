import pytest

from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


def _pair(chunk_id: str, vector_rank: int, summary_rank: int | None = None) -> dict:
    return {
        "chunk": {"id": chunk_id, "text": chunk_id, "importance_weight": 0.5},
        "chunk_id": chunk_id,
        "vector_rank": vector_rank,
        "summary_rank": summary_rank,
    }


def test_min_score_filters_on_final_fused_score():
    pairs = [_pair("two-lane", 0, 0), _pair("one-lane", 1)]

    ranked = rank_chunk_summary_pairs(
        pairs,
        limit=2,
        use_importance_weight=False,
        min_score=0.05,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["two-lane"]


def test_omitted_min_score_preserves_existing_ranking():
    pairs = [_pair("first", 0), _pair("second", 1)]

    ranked = rank_chunk_summary_pairs(pairs, limit=2, use_importance_weight=False)

    assert [pair["chunk_id"] for pair in ranked] == ["first", "second"]


@pytest.mark.parametrize("value", [-0.1, True])
def test_hybrid_retriever_rejects_invalid_min_score(value):
    with pytest.raises(ValueError, match="min_score"):
        HybridRetriever(min_score=value)
