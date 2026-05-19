from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .bucket_assignment import assign_items_to_buckets, create_buckets_for_level
from .graph_bucketing import place_graph_summaries_incrementally, rebuild_graph_buckets_for_level
from .models import BucketAssignment, SummaryNode

BucketingStrategyName = Literal["vector", "graph"]


@dataclass
class BucketingStrategyContext:
    vector_engine: Any
    source_collection: str
    placement_distance_threshold: float
    min_overlap: float
    entities_by_summary_id: dict[str, set[str]] = field(default_factory=dict)
    idf_weights: dict[str, float] = field(default_factory=dict)


class BucketingStrategy(Protocol):
    async def assign(
        self,
        *,
        items: list[SummaryNode],
        items_for_cluster: list[SummaryNode],
        existing_buckets: list[SummaryNode],
        children_by_id: dict[str, SummaryNode],
        level: int,
        dataset_id: str,
        max_bucket_size: int,
        strategy_context: BucketingStrategyContext,
    ) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]: ...


class VectorBucketingStrategy:
    async def assign(
        self,
        *,
        items: list[SummaryNode],
        items_for_cluster: list[SummaryNode],
        existing_buckets: list[SummaryNode],
        children_by_id: dict[str, SummaryNode],
        level: int,
        dataset_id: str,
        max_bucket_size: int,
        strategy_context: BucketingStrategyContext,
    ) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
        if existing_buckets or level == 0:
            return await assign_items_to_buckets(
                items,
                existing_buckets,
                level,
                dataset_id,
                strategy_context.vector_engine,
                strategy_context.source_collection,
                max_bucket_size,
                strategy_context.placement_distance_threshold,
            )

        return await create_buckets_for_level(
            items_for_cluster,
            level,
            dataset_id,
            strategy_context.vector_engine,
            strategy_context.source_collection,
            max_bucket_size,
        )


class GraphBucketingStrategy:
    async def assign(
        self,
        *,
        items: list[SummaryNode],
        items_for_cluster: list[SummaryNode],
        existing_buckets: list[SummaryNode],
        children_by_id: dict[str, SummaryNode],
        level: int,
        dataset_id: str,
        max_bucket_size: int,
        strategy_context: BucketingStrategyContext,
    ) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
        if level != 0:
            raise ValueError("Graph bucketing is only supported for level 0.")
        if existing_buckets:
            return place_graph_summaries_incrementally(
                items,
                existing_buckets,
                strategy_context.entities_by_summary_id,
                strategy_context.idf_weights,
                dataset_id=dataset_id,
                level=level,
                max_bucket_size=max_bucket_size,
                min_overlap=strategy_context.min_overlap,
            )

        return rebuild_graph_buckets_for_level(
            items_for_cluster,
            strategy_context.entities_by_summary_id,
            strategy_context.idf_weights,
            dataset_id=dataset_id,
            level=level,
            max_bucket_size=max_bucket_size,
            min_overlap=strategy_context.min_overlap,
        )
