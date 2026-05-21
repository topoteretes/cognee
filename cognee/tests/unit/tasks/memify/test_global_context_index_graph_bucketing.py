import pytest

from cognee.tasks.memify.global_context_index.bucketing.graph.placement import (
    place_graph_summaries_incrementally,
    rebuild_graph_buckets_for_level,
    validate_graph_buckets_can_be_extended,
    validate_vector_buckets_can_be_extended,
)
from cognee.tasks.memify.global_context_index.ids import create_bucket_id
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


def _bucket(
    bucket_id: str,
    child_ids: set[str],
    graph_bucket_entity_ids: set[str] | None,
) -> SummaryNode:
    return SummaryNode(
        id=bucket_id,
        text=f"{bucket_id} text",
        type="GlobalContextSummary",
        level=0,
        child_ids=child_ids,
        graph_bucket_entity_ids=graph_bucket_entity_ids,
    )


def _incremental(
    new_summary_ids: list[str],
    existing_buckets: list[SummaryNode],
    entities_by_summary_id: dict[str, set[str]],
    idf_weights: dict[str, float],
    *,
    max_bucket_size: int = 2,
    min_overlap: float = 0.1,
):
    return place_graph_summaries_incrementally(
        [_summary(summary_id) for summary_id in new_summary_ids],
        existing_buckets,
        entities_by_summary_id,
        idf_weights,
        dataset_id="dataset-1",
        level=0,
        max_bucket_size=max_bucket_size,
        min_overlap=min_overlap,
    )


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
    assert {(assignment.child_id, assignment.parent_id) for assignment in assignments} == {
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


def test_validate_vector_buckets_can_be_extended_rejects_graph_built_buckets():
    with pytest.raises(ValueError, match="cannot extend graph-built"):
        validate_vector_buckets_can_be_extended([_bucket("graph-bucket", {"s1"}, {"alice"})])


def test_graph_incremental_places_summary_into_matching_existing_bucket():
    existing_bucket = _bucket("bucket-alice", {"s1"}, {"alice"})

    buckets, assignments = _incremental(
        ["s2"],
        [existing_bucket],
        {"s1": {"alice"}, "s2": {"alice"}},
        {"alice": 1.0},
        max_bucket_size=3,
    )

    assert buckets == {"bucket-alice": existing_bucket}
    assert existing_bucket.child_ids == {"s1", "s2"}
    assert existing_bucket.graph_bucket_entity_ids == {"alice"}
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s2", "bucket-alice")
    ]


def test_graph_incremental_creates_new_bucket_when_overlap_is_below_threshold():
    existing_bucket = _bucket("bucket-rare", {"s1"}, {"rare", "common"})

    buckets, assignments = _incremental(
        ["s2"],
        [existing_bucket],
        {"s1": {"rare", "common"}, "s2": {"common"}},
        {"rare": 0.95, "common": 0.05},
        min_overlap=0.1,
    )

    new_bucket_id = str(create_bucket_id("dataset-1", 0, ["s2"]))
    assert set(buckets) == {new_bucket_id}
    assert buckets[new_bucket_id].child_ids == {"s2"}
    assert buckets[new_bucket_id].graph_bucket_entity_ids == {"common"}
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s2", new_bucket_id)
    ]
    assert existing_bucket.child_ids == {"s1"}


def test_graph_incremental_reuses_misc_bucket_for_no_effective_entity_summary():
    misc_bucket = _bucket("misc-bucket", {"s1"}, set())

    buckets, assignments = _incremental(
        ["s2"],
        [misc_bucket],
        {"s1": set(), "s2": {"standup"}},
        {"standup": 0.0},
        max_bucket_size=2,
        min_overlap=0,
    )

    assert buckets == {"misc-bucket": misc_bucket}
    assert misc_bucket.child_ids == {"s1", "s2"}
    assert misc_bucket.graph_bucket_entity_ids == set()
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s2", "misc-bucket")
    ]


def test_graph_incremental_updates_index_after_creating_bucket():
    buckets, assignments = _incremental(
        ["s1", "s2"],
        [],
        {"s1": {"alice"}, "s2": {"alice"}},
        {"alice": 1.0},
        max_bucket_size=2,
    )

    bucket_id = str(create_bucket_id("dataset-1", 0, ["s1"]))
    assert set(buckets) == {bucket_id}
    assert buckets[bucket_id].child_ids == {"s1", "s2"}
    assert buckets[bucket_id].graph_bucket_entity_ids == {"alice"}
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s1", bucket_id),
        ("s2", bucket_id),
    ]


def test_graph_incremental_tie_breaks_by_bucket_size_then_id():
    larger_bucket = _bucket("bucket-a", {"s1", "s2"}, {"shared"})
    smaller_bucket = _bucket("bucket-b", {"s3"}, {"shared"})

    _, assignments = _incremental(
        ["new"],
        [larger_bucket, smaller_bucket],
        {
            "s1": {"shared"},
            "s2": {"shared"},
            "s3": {"shared"},
            "new": {"shared"},
        },
        {"shared": 1.0},
        max_bucket_size=3,
    )

    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("new", "bucket-b")
    ]


def test_graph_incremental_tie_breaks_equal_size_by_bucket_id():
    bucket_b = _bucket("bucket-b", {"s1"}, {"shared"})
    bucket_a = _bucket("bucket-a", {"s2"}, {"shared"})

    _, assignments = _incremental(
        ["new"],
        [bucket_b, bucket_a],
        {
            "s1": {"shared"},
            "s2": {"shared"},
            "new": {"shared"},
        },
        {"shared": 1.0},
        max_bucket_size=2,
    )

    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("new", "bucket-a")
    ]


def test_graph_incremental_normalizes_misc_bucket_before_placement():
    bucket = _bucket("bucket-promoted", {"s1"}, set())

    buckets, assignments = _incremental(
        ["s2"],
        [bucket],
        {"s1": {"alice"}, "s2": {"alice"}},
        {"alice": 1.0},
        max_bucket_size=2,
    )

    assert buckets == {"bucket-promoted": bucket}
    assert bucket.child_ids == {"s1", "s2"}
    assert bucket.graph_bucket_entity_ids == {"alice"}
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s2", "bucket-promoted")
    ]


def test_graph_incremental_demotes_zero_weight_bucket_before_misc_reuse():
    bucket = _bucket("bucket-demoted", {"s1"}, {"standup"})

    buckets, assignments = _incremental(
        ["s2"],
        [bucket],
        {"s1": {"standup"}, "s2": set()},
        {"standup": 0.0},
        max_bucket_size=2,
        min_overlap=0,
    )

    assert buckets == {"bucket-demoted": bucket}
    assert bucket.child_ids == {"s1", "s2"}
    assert bucket.graph_bucket_entity_ids == set()
    assert [(assignment.child_id, assignment.parent_id) for assignment in assignments] == [
        ("s2", "bucket-demoted")
    ]
