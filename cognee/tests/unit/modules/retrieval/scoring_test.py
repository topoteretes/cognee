from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.retrieval.utils.scoring import attach_scores, filter_by_max_distance
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _scored(score, payload=None):
    return ScoredResult(id=uuid4(), score=score, payload=payload)


def test_filter_by_max_distance_drops_results_above_cutoff():
    results = [_scored(0.1), _scored(0.5), _scored(0.9)]
    kept = filter_by_max_distance(results, 0.5)
    assert [r.score for r in kept] == [0.1, 0.5]


def test_filter_by_max_distance_keeps_boundary_value():
    results = [_scored(0.5)]
    assert filter_by_max_distance(results, 0.5) == results


def test_filter_by_max_distance_none_returns_all_unchanged():
    results = [_scored(0.1), _scored(0.9)]
    assert filter_by_max_distance(results, None) is results


def test_attach_scores_adds_distance_to_each_payload():
    results = [_scored(0.2, {"text": "a"}), _scored(0.4, {"text": "b"})]
    assert attach_scores(results) == [
        {"text": "a", "score": 0.2},
        {"text": "b", "score": 0.4},
    ]


def test_attach_scores_handles_missing_payload():
    assert attach_scores([_scored(0.3, None)]) == [{"score": 0.3}]


def test_attach_scores_preserves_all_payload_fields():
    results = [_scored(0.2, {"text": "a", "chunk_index": 3, "metadata": {"k": "v"}})]
    assert attach_scores(results) == [
        {"text": "a", "chunk_index": 3, "metadata": {"k": "v"}, "score": 0.2}
    ]


@pytest.mark.parametrize("search_type", [SearchType.CHUNKS, SearchType.SUMMARIES])
def test_attached_score_reaches_searchresultitem(search_type):
    """End-to-end: attach_scores output is read by the SearchResultItem.score consumer.

    This is the whole point of the feature -- the score key it adds is exactly the
    key normalize_search_payload already reads to populate SearchResultItem.score.
    """
    results = [_scored(0.2, {"text": "a"}), _scored(0.6, {"text": "b"})]
    payload = SearchResultPayload(completion=attach_scores(results), search_type=search_type)

    items = normalize_search_payload(payload)

    assert [item.score for item in items] == [0.2, 0.6]
    assert items[0].raw == {"text": "a", "score": 0.2}
