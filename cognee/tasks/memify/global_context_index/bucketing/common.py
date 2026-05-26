from __future__ import annotations

from collections.abc import Iterable

from ..ids import create_bucket_id
from ..models import BucketAssignment, SummaryNode


def create_bucket_node(
    child_ids: Iterable[str],
    dataset_id: str,
    level: int,
    *,
    graph_bucket_entity_ids: set[str] | None = None,
) -> SummaryNode:
    child_id_set = set(child_ids)
    return SummaryNode(
        id=str(create_bucket_id(dataset_id, level, list(child_id_set))),
        text="",
        type="GlobalContextSummary",
        level=level,
        is_root=False,
        dataset_id=dataset_id,
        child_ids=child_id_set,
        graph_bucket_entity_ids=graph_bucket_entity_ids,
    )


def add_child_to_bucket(
    child: SummaryNode,
    bucket: SummaryNode,
    assignments: list[BucketAssignment],
) -> None:
    bucket.child_ids.add(child.id)
    child.global_context_bucket_id = bucket.id
    record_bucket_assignment(assignments, child.id, bucket.id)


def record_bucket_assignment(
    assignments: list[BucketAssignment],
    child_id: str,
    bucket_id: str,
) -> None:
    assignments.append(BucketAssignment(child_id=child_id, parent_id=bucket_id))


def mark_bucket_for_persistence(
    buckets_to_persist: dict[str, SummaryNode],
    bucket: SummaryNode,
) -> None:
    buckets_to_persist[bucket.id] = bucket
