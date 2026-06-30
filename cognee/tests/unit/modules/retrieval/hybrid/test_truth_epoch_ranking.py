from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _pair(chunk_id: str, rank: int) -> dict:
    return {
        "chunk": {"id": chunk_id, "text": chunk_id, "importance_weight": 0.5},
        "chunk_id": chunk_id,
        "bm25_rank": rank,
        "vector_rank": None,
        "summary_rank": None,
    }


def test_truth_weight_only_applies_to_current_epoch_vectors():
    ranked = rank_chunk_summary_pairs(
        [_pair("stale", 0), _pair("current", 1)],
        limit=2,
        use_importance_weight=False,
        use_truth_weight=True,
        q_coords=[1.0],
        truth_state_by_id={
            "stale": {"truth_alignment": [1.0], "truth_epoch": 1},
            "current": {"truth_alignment": [1.0], "truth_epoch": 2},
        },
        current_truth_epoch=2,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["current", "stale"]


def test_truth_weight_ignores_all_vectors_when_epoch_is_unknown():
    ranked = rank_chunk_summary_pairs(
        [_pair("first", 0), _pair("second", 1)],
        limit=2,
        use_importance_weight=False,
        use_truth_weight=True,
        q_coords=[1.0],
        truth_state_by_id={
            "second": {"truth_alignment": [1.0], "truth_epoch": 2},
        },
        current_truth_epoch=None,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["first", "second"]
