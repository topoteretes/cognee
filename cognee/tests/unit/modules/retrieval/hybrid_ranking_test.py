import pytest

from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _pair(
    chunk_id: str,
    *,
    bm25_rank=None,
    vector_rank=None,
    summary_rank=None,
    importance_weight=0.5,
    **scores,
):
    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_id,
        "summary_id": None,
        "summary_text": None,
        "chunk": {
            "id": chunk_id,
            "text": chunk_id,
            "importance_weight": importance_weight,
        },
        "bm25_rank": bm25_rank,
        "vector_rank": vector_rank,
        "summary_rank": summary_rank,
        **scores,
    }


def test_summary_is_supporting_evidence_by_default_and_attribution_is_exposed():
    lexical = _pair("lexical", bm25_rank=0)
    derived = _pair("derived", summary_rank=0)

    ranked = rank_chunk_summary_pairs(
        [derived, lexical], limit=2, use_importance_weight=False, rrf_k=30
    )

    assert [pair["chunk_id"] for pair in ranked] == ["lexical", "derived"]
    assert ranked[0]["retrieval_score"] == pytest.approx(1 / 31)
    assert ranked[1]["retrieval_score"] == pytest.approx(0.5 / 31)
    assert ranked[1]["retrieval_channels"] == [
        {
            "channel": "summary",
            "family": "semantic",
            "rank": 0,
            "weight": 0.5,
            "contribution": pytest.approx(0.5 / 31),
        }
    ]


def test_chunk_and_derived_summary_are_one_semantic_vote_family():
    exact = _pair("exact", bm25_rank=0)
    correlated = _pair("correlated", vector_rank=10, summary_rank=10)

    ranked = rank_chunk_summary_pairs(
        [correlated, exact], limit=2, use_importance_weight=False, rrf_k=50
    )

    assert [pair["chunk_id"] for pair in ranked] == ["exact", "correlated"]
    assert ranked[1]["rrf_score"] == pytest.approx(1 / 61)


def test_channel_weights_are_configurable_without_changing_callers():
    lexical = _pair("lexical", bm25_rank=0)
    summary = _pair("summary", summary_rank=0)

    ranked = rank_chunk_summary_pairs(
        [lexical, summary],
        limit=2,
        use_importance_weight=False,
        channel_weights={"bm25": 0.25, "summary": 2.0},
        rrf_k=30,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["summary", "lexical"]


def test_importance_cannot_overpower_primary_relevance_and_breaks_exact_ties():
    first = _pair("first", vector_rank=0, importance_weight=0.0)
    second = _pair("second", vector_rank=1, importance_weight=1.0)
    tied_low = _pair("tied-low", bm25_rank=0, importance_weight=0.0)
    tied_high = _pair("tied-high", bm25_rank=0, importance_weight=1.0)

    ranked = rank_chunk_summary_pairs(
        [second, first, tied_low, tied_high],
        limit=4,
        use_importance_weight=True,
        rrf_k=30,
    )

    assert ranked.index(
        next(pair for pair in ranked if pair["chunk_id"] == "first")
    ) < ranked.index(next(pair for pair in ranked if pair["chunk_id"] == "second"))
    assert ranked.index(
        next(pair for pair in ranked if pair["chunk_id"] == "tied-high")
    ) < ranked.index(next(pair for pair in ranked if pair["chunk_id"] == "tied-low"))


def test_native_scores_are_retained_for_observability_but_not_mixed_uncalibrated():
    pair = _pair("chunk", bm25_rank=0, bm25_score=12.5)

    ranked = rank_chunk_summary_pairs([pair], limit=1, use_importance_weight=False)

    assert ranked[0]["retrieval_channels"][0]["native_score"] == 12.5
