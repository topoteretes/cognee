from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _pair(chunk_id: str, rank: int, importance: float = 0.5) -> dict:
    return {
        "chunk": {"id": chunk_id, "text": chunk_id, "importance_weight": importance},
        "chunk_id": chunk_id,
        "bm25_rank": rank,
        "vector_rank": None,
        "summary_rank": None,
    }


def test_personal_weights_reorder_pairs():
    ranked = rank_chunk_summary_pairs(
        [_pair("a", 0), _pair("b", 1)],
        limit=2,
        use_importance_weight=False,
        personal_weights={"a": 0.05, "b": 0.95},
        personal_influence=0.3,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["b", "a"]


def test_personal_weights_compose_with_importance_and_truth_factors():
    ranked = rank_chunk_summary_pairs(
        [_pair("a", 0, importance=0.9), _pair("b", 1, importance=0.1)],
        limit=2,
        use_importance_weight=True,
        use_truth_weight=True,
        q_coords=[1.0],
        truth_state_by_id={
            "a": {"truth_alignment": [1.0], "truth_epoch": 2},
            "b": {"truth_alignment": [0.5], "truth_epoch": 2},
        },
        current_truth_epoch=2,
        personal_weights={"a": 0.05, "b": 0.95},
        personal_influence=0.3,
    )

    assert {pair["chunk_id"] for pair in ranked} == {"a", "b"}


def test_empty_personal_weight_map_is_byte_identical_to_baseline():
    pairs = [_pair("a", 1), _pair("b", 0), _pair("c", 2)]

    baseline = rank_chunk_summary_pairs(pairs, limit=3, use_importance_weight=True)
    with_empty_map = rank_chunk_summary_pairs(
        pairs,
        limit=3,
        use_importance_weight=True,
        personal_weights={},
        personal_influence=0.3,
    )
    with_none_map = rank_chunk_summary_pairs(
        pairs,
        limit=3,
        use_importance_weight=True,
        personal_weights=None,
        personal_influence=0.3,
    )

    assert with_empty_map == baseline
    assert with_none_map == baseline


def test_chunks_without_a_weight_keep_baseline_order():
    pairs = [_pair("a", 0), _pair("b", 1), _pair("c", 2)]

    ranked = rank_chunk_summary_pairs(
        pairs,
        limit=3,
        use_importance_weight=False,
        personal_weights={"unrelated": 0.95},
        personal_influence=0.3,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["a", "b", "c"]


def test_neutral_weight_is_an_exact_no_op():
    pairs = [_pair("a", 0), _pair("b", 1)]

    baseline = rank_chunk_summary_pairs(pairs, limit=2, use_importance_weight=False)
    ranked = rank_chunk_summary_pairs(
        pairs,
        limit=2,
        use_importance_weight=False,
        personal_weights={"a": 0.5, "b": 0.5},
        personal_influence=0.3,
    )

    assert ranked == baseline
