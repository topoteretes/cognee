"""Tests for choose_best_bucket placement scoring.

choose_best_bucket averages a bucket's child distances to score it. A bucket with
no children has an empty distance list, which slipped past the
`len(distances) != len(child_ids)` guard (0 == 0) and hit `sum([]) / len([])`,
raising ZeroDivisionError. Empty buckets are now skipped.
"""

from __future__ import annotations

from cognee.tasks.memify.global_context_index.bucketing.vector.placement import (
    choose_best_bucket,
)
from cognee.tasks.memify.global_context_index.models import SummaryNode


def _bucket(bucket_id: str, child_ids: set[str]) -> SummaryNode:
    return SummaryNode(id=bucket_id, text="", type="bucket", child_ids=set(child_ids))


def test_empty_bucket_is_skipped_not_crashing():
    empty = _bucket("b1", set())
    result = choose_best_bucket(
        {"b1": []},
        {"b1": empty},
        max_bucket_size=10,
        placement_distance_threshold=1.0,
    )
    assert result is None


def test_empty_bucket_skipped_but_valid_bucket_still_chosen():
    empty = _bucket("empty", set())
    valid = _bucket("valid", {"c1", "c2"})
    result = choose_best_bucket(
        {"empty": [], "valid": [0.1, 0.2]},
        {"empty": empty, "valid": valid},
        max_bucket_size=10,
        placement_distance_threshold=1.0,
    )
    assert result is valid


def test_lower_mean_distance_bucket_wins():
    near = _bucket("near", {"c1"})
    far = _bucket("far", {"c2"})
    result = choose_best_bucket(
        {"near": [0.1], "far": [0.9]},
        {"near": near, "far": far},
        max_bucket_size=10,
        placement_distance_threshold=1.0,
    )
    assert result is near


def test_bucket_over_distance_threshold_is_skipped():
    far = _bucket("far", {"c1"})
    result = choose_best_bucket(
        {"far": [0.95]},
        {"far": far},
        max_bucket_size=10,
        placement_distance_threshold=0.5,
    )
    assert result is None
