from __future__ import annotations

import math
from numbers import Real
from typing import Any

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import GlobalContextSummary

from .build_context_index import build_context_index, group_buckets_by_level
from .graph_input import dataset_id_from_context, load_context_index_input
from .models import GlobalContextIndexInput, SummaryNode
from .persistence import delete_context_index_nodes, persist_context_index_edges

logger = get_logger("global_context_index")


def validate_global_context_index_config(
    max_bucket_size: int,
    placement_distance_threshold: float,
) -> None:
    if isinstance(max_bucket_size, bool) or not isinstance(max_bucket_size, int):
        raise ValueError("max_bucket_size must be an integer.")
    if max_bucket_size < 2:
        raise ValueError("max_bucket_size must be at least 2.")

    if isinstance(placement_distance_threshold, bool) or not isinstance(
        placement_distance_threshold, Real
    ):
        raise ValueError("placement_distance_threshold must be a finite number.")
    if placement_distance_threshold < 0 or not math.isfinite(placement_distance_threshold):
        raise ValueError("placement_distance_threshold must be a finite non-negative number.")


async def reset_context_index_for_rebuild(
    context_input: GlobalContextIndexInput,
    text_summaries: list[SummaryNode],
    unified_engine: Any,
) -> None:
    all_bucket_ids = [bucket.id for bucket in context_input.buckets]
    if context_input.root:
        all_bucket_ids.append(context_input.root.id)
    await delete_context_index_nodes(unified_engine, all_bucket_ids)
    context_input.buckets = []
    context_input.root = None
    for summary in text_summaries:
        summary.global_context_bucket_id = None


def select_new_context_index_items(
    text_summaries: list[SummaryNode],
    rebuild: bool,
) -> list[SummaryNode]:
    if rebuild:
        return text_summaries
    return [summary for summary in text_summaries if not summary.global_context_bucket_id]


@task_summary("Updated {n} global context index node(s)")
async def update_global_context_index(
    data: Any = None,
    rebuild: bool = False,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    ctx: PipelineContext | None = None,
) -> list[GlobalContextSummary]:
    """
    Build or incrementally extend the global context index above a dataset's
    TextSummaries: level-0 buckets, any upper levels needed, and the dataset
    root.
    """
    validate_global_context_index_config(max_bucket_size, placement_distance_threshold)

    dataset_id = dataset_id_from_context(ctx)
    context_input = await load_context_index_input(data, dataset_id, ctx)
    text_summaries = [summary for summary in context_input.text_summaries if summary.text]

    unified_engine = None

    if rebuild:
        unified_engine = await get_unified_engine()
        await reset_context_index_for_rebuild(context_input, text_summaries, unified_engine)

    if not text_summaries:
        return []

    if unified_engine is None:
        unified_engine = await get_unified_engine()

    new_summaries = select_new_context_index_items(text_summaries, rebuild)

    if not new_summaries:
        logger.info("No new TextSummary nodes to place in global context index.")
        return []

    context_datapoints, assignments = await build_context_index(
        new_text_summaries=new_summaries,
        text_summaries_all=text_summaries,
        buckets_by_level=group_buckets_by_level(context_input.buckets),
        existing_root=context_input.root,
        dataset_id=dataset_id,
        vector_engine=unified_engine.vector,
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        ctx=ctx,
    )

    await persist_context_index_edges(assignments, unified_engine)

    return context_datapoints
