import pytest

from cognee.tasks.memify.global_context_index.bucket_assignment import create_bucket_id
from cognee.tasks.memify.global_context_index.graph_bucketing import (
    rebuild_graph_buckets_for_level,
    validate_graph_buckets_can_be_extended,
)
from cognee.tasks.memify.global_context_index.models import SummaryNode


def _summary(summary_id: str) -> SummaryNode:
    return SummaryNode(id=summary_id, text=f"{summary_id} text", type="TextSummary")


def _rebuild(
    summary_ids: list[str],
    entities_by_summary_id: dict[str, set[str]],
    idf_weights: dict[str, float],
    *,
    max_bucket_size: int = 2,
    min_overlap: float = 0.1,
):
    return rebuild_graph_buckets_for_level(
        [_summary(summary_id) for summary_id in summary_ids],
        entities_by_summary_id,
        idf_weights,
        dataset_id="dataset-1",
        level=0,
        max_bucket_size=max_bucket_size,
        min_overlap=min_overlap,
    )


def _bucket_child_sets(buckets: dict[str, SummaryNode]) -> list[set[str]]:
    return [bucket.child_ids for bucket in buckets.values()]


def _bucket_entity_sets(buckets: dict[str, SummaryNode]) -> list[set[str] | None]:
    return [bucket.graph_bucket_entity_ids for bucket in buckets.values()]


def test_graph_rebuild_groups_by_weighted_overlap_and_populates_entity_state():
    buckets, assignments = _rebuild(
        ["s1", "s2", "s3", "s4"],
        {
            "s1": {"alice", "project-x"},
            "s2": {"alice"},
            "s3": {"bob"},
            "s4": {"bob"},
        },
        {"alice": 1.0, "project-x": 1.0, "bob": 1.0},
    )

    assert _bucket_child_sets(buckets) == [{"s1", "s2"}, {"s3", "s4"}]
    assert _bucket_entity_sets(buckets) == [{"alice", "project-x"}, {"bob"}]
    assert {(assignment.summary_id, assignment.bucket_id) for assignment in assignments} == {
        (child_id, bucket.id) for bucket in buckets.values() for child_id in bucket.child_ids
    }


def test_graph_rebuild_uses_weighted_seed_order():
    buckets, _ = _rebuild(
        ["plain", "rich", "other"],
        {
            "plain": {"alice"},
            "rich": {"alice", "project-x"},
            "other": {"bob"},
        },
        {"alice": 1.0, "project-x": 1.0, "bob": 1.0},
        max_bucket_size=1,
    )

    assert [bucket.child_ids for bucket in buckets.values()] == [{"rich"}, {"other"}, {"plain"}]


def test_graph_rebuild_respects_weighted_jaccard_cutoff():
    buckets, _ = _rebuild(
        ["s1", "s2"],
        {
            "s1": {"rare", "common"},
            "s2": {"common"},
        },
        {"rare": 0.95, "common": 0.05},
        min_overlap=0.1,
    )

    assert _bucket_child_sets(buckets) == [{"s1"}, {"s2"}]


def test_graph_rebuild_tie_breaks_candidates_by_summary_id():
    buckets, _ = _rebuild(
        ["seed", "candidate-b", "candidate-c"],
        {
            "seed": {"shared", "seed-a", "seed-b"},
            "candidate-b": {"shared", "b"},
            "candidate-c": {"shared", "c"},
        },
        {"shared": 1.0, "seed-a": 1.0, "seed-b": 1.0, "b": 1.0, "c": 1.0},
        max_bucket_size=2,
    )

    assert next(iter(buckets.values())).child_ids == {"seed", "candidate-b"}


def test_graph_rebuild_places_no_entity_and_zero_weight_summaries_in_misc_buckets():
    buckets, assignments = _rebuild(
        ["missing", "no-entities", "zero-weight"],
        {
            "no-entities": set(),
            "zero-weight": {"standup"},
        },
        {"standup": 0.0},
        max_bucket_size=2,
        min_overlap=0,
    )

    assert _bucket_child_sets(buckets) == [{"missing", "no-entities"}, {"zero-weight"}]
    assert _bucket_entity_sets(buckets) == [set(), set()]
    assert len(assignments) == 3


def test_graph_rebuild_min_overlap_zero_keeps_zero_weight_only_summaries_misc():
    buckets, _ = _rebuild(
        ["entity", "zero-weight"],
        {
            "entity": {"alice"},
            "zero-weight": {"standup"},
        },
        {"alice": 1.0, "standup": 0.0},
        min_overlap=0,
    )

    assert _bucket_child_sets(buckets) == [{"entity"}, {"zero-weight"}]
    assert _bucket_entity_sets(buckets) == [{"alice"}, set()]


def test_graph_rebuild_min_overlap_zero_admits_positive_candidates_with_zero_score():
    buckets, _ = _rebuild(
        ["s1", "s2"],
        {
            "s1": {"standup", "alice"},
            "s2": {"standup", "bob"},
        },
        {"standup": 0.0, "alice": 1.0, "bob": 1.0},
        min_overlap=0,
    )

    assert _bucket_child_sets(buckets) == [{"s1", "s2"}]


def test_graph_rebuild_bucket_ids_are_deterministic():
    first_buckets, first_assignments = _rebuild(
        ["s1", "s2", "s3"],
        {
            "s1": {"alice"},
            "s2": {"alice"},
            "s3": {"bob"},
        },
        {"alice": 1.0, "bob": 1.0},
    )
    second_buckets, second_assignments = _rebuild(
        ["s1", "s2", "s3"],
        {
            "s1": {"alice"},
            "s2": {"alice"},
            "s3": {"bob"},
        },
        {"alice": 1.0, "bob": 1.0},
    )

    assert first_buckets == second_buckets
    assert first_assignments == second_assignments
    assert set(first_buckets) == {
        str(create_bucket_id("dataset-1", 0, list(bucket.child_ids)))
        for bucket in first_buckets.values()
    }


def test_graph_rebuild_does_not_mutate_input_summaries():
    summaries = [_summary("s1"), _summary("s2")]

    rebuild_graph_buckets_for_level(
        summaries,
        {"s1": {"alice"}, "s2": {"alice"}},
        {"alice": 1.0},
        dataset_id="dataset-1",
        level=0,
        max_bucket_size=2,
        min_overlap=0.1,
    )

    assert [summary.global_context_bucket_id for summary in summaries] == [None, None]
    assert [summary.child_ids for summary in summaries] == [set(), set()]


def test_validate_graph_buckets_can_be_extended_rejects_missing_level_zero_state():
    bucket = SummaryNode(
        id="bucket-1",
        text="bucket",
        type="GlobalContextSummary",
        level=0,
        graph_bucket_entity_ids=None,
    )

    with pytest.raises(ValueError, match="Use rebuild=True"):
        validate_graph_buckets_can_be_extended([bucket])


def test_validate_graph_buckets_can_be_extended_accepts_empty_graph_state():
    validate_graph_buckets_can_be_extended(
        [
            SummaryNode(
                id="misc-bucket",
                text="misc",
                type="GlobalContextSummary",
                level=0,
                graph_bucket_entity_ids=set(),
            ),
            SummaryNode(
                id="upper-bucket",
                text="upper",
                type="GlobalContextSummary",
                level=1,
                graph_bucket_entity_ids=None,
            ),
        ]
    )
