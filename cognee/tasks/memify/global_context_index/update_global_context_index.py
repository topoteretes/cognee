from __future__ import annotations

import math
from numbers import Real
from typing import Any

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.graph.methods.get_global_context_graph_inputs import get_dataset_text_summary_ids
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import GlobalContextSummary

from .build_context_index import build_context_index, group_buckets_by_level
from .bucketing_strategy import BucketingStrategyName
from .graph_bucketing import (
    validate_graph_buckets_can_be_extended,
    validate_vector_buckets_can_be_extended,
)
from .graph_input import dataset_id_from_context, load_context_index_input
from .graph_providers import GlobalContextGraphInput, load_global_context_graph_input
from .models import GlobalContextIndexInput, SummaryNode
from .persistence import delete_context_index_nodes, persist_context_index_edges

logger = get_logger("global_context_index")


def validate_global_context_index_config(
    max_bucket_size: int,
    placement_distance_threshold: float,
    bucketing_strategy: BucketingStrategyName,
    min_overlap: float,
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

    if bucketing_strategy not in ("vector", "graph"):
        raise ValueError('bucketing_strategy must be "vector" or "graph".')

    if isinstance(min_overlap, bool) or not isinstance(min_overlap, Real):
        raise ValueError("min_overlap must be a finite number in [0.0, 1.0].")
    if not math.isfinite(min_overlap) or min_overlap < 0 or min_overlap > 1:
        raise ValueError("min_overlap must be a finite number in [0.0, 1.0].")


def validate_graph_rebuild_input(graph_input: GlobalContextGraphInput) -> None:
    missing = graph_input.summary_entities.missing_made_from_summary_ids
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            'bucketing_strategy="graph" requires every TextSummary to have a made_from '
            f"chunk edge. Missing made_from for {len(missing)} summary id(s): {sample}{suffix}"
        )


async def filter_graph_dataset_text_summaries(
    dataset_id: str,
    text_summaries: list[SummaryNode],
) -> list[SummaryNode]:
    dataset_summary_ids = set(await get_dataset_text_summary_ids(dataset_id))
    summaries_by_id = {summary.id: summary for summary in text_summaries}
    missing_summary_ids = dataset_summary_ids - set(summaries_by_id)

    if missing_summary_ids:
        sample = ", ".join(sorted(missing_summary_ids)[:5])
        suffix = "..." if len(missing_summary_ids) > 5 else ""
        raise ValueError(
            'bucketing_strategy="graph" could not load graph TextSummary node(s) '
            f"for {len(missing_summary_ids)} dataset summary id(s): {sample}{suffix}"
        )

    return [summaries_by_id[summary_id] for summary_id in sorted(dataset_summary_ids)]


async def filter_text_summaries_for_strategy(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
    text_summaries: list[SummaryNode],
) -> list[SummaryNode]:
    if bucketing_strategy != "graph":
        return text_summaries
    return await filter_graph_dataset_text_summaries(dataset_id, text_summaries)


async def load_validated_graph_input(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
    text_summaries: list[SummaryNode],
) -> GlobalContextGraphInput | None:
    if bucketing_strategy != "graph" or not text_summaries:
        return None

    graph_input = await load_global_context_graph_input(
        dataset_id,
        [summary.id for summary in text_summaries],
    )
    validate_graph_rebuild_input(graph_input)
    return graph_input


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


def validate_existing_buckets_can_be_extended(
    bucketing_strategy: BucketingStrategyName,
    existing_buckets: list[SummaryNode],
) -> None:
    if bucketing_strategy == "graph":
        validate_graph_buckets_can_be_extended(existing_buckets)
        return

    validate_vector_buckets_can_be_extended(existing_buckets)


@task_summary("Updated {n} global context index node(s)")
async def update_global_context_index(
    data: Any = None,
    rebuild: bool = False,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.1,
    ctx: PipelineContext | None = None,
) -> list[GlobalContextSummary]:
    """
    Build or incrementally extend the global context index above a dataset's
    TextSummaries: level-0 buckets, any upper levels needed, and the dataset
    root.

    ``bucketing_strategy="vector"`` is the stable default and uses
    ``placement_distance_threshold``. ``bucketing_strategy="graph"`` is
    experimental, uses ``min_overlap``, and only applies to level 0. Existing
    vector-built buckets cannot be extended by graph incremental mode, and
    existing graph-built buckets cannot be extended by vector incremental mode;
    use ``rebuild=True`` to switch strategies.
    """
    validate_global_context_index_config(
        max_bucket_size,
        placement_distance_threshold,
        bucketing_strategy,
        min_overlap,
    )
    dataset_id = dataset_id_from_context(ctx)
    if bucketing_strategy == "graph" and not dataset_id:
        raise ValueError('bucketing_strategy="graph" requires a dataset context.')

    context_input = await load_context_index_input(data, dataset_id, ctx)
    text_summaries = [summary for summary in context_input.text_summaries if summary.text]
    text_summaries = await filter_text_summaries_for_strategy(
        bucketing_strategy,
        dataset_id,
        text_summaries,
    )

    unified_engine = None
    graph_input = None

    if rebuild:
        graph_input = await load_validated_graph_input(
            bucketing_strategy,
            dataset_id,
            text_summaries,
        )
        unified_engine = await get_unified_engine()
        await reset_context_index_for_rebuild(context_input, text_summaries, unified_engine)
    else:
        validate_existing_buckets_can_be_extended(bucketing_strategy, context_input.buckets)

    if not text_summaries:
        return []

    if unified_engine is None:
        unified_engine = await get_unified_engine()

    new_summaries = select_new_context_index_items(text_summaries, rebuild)

    if not new_summaries:
        logger.info("No new TextSummary nodes to place in global context index.")
        return []

    entities_by_summary_id = context_input.entities_by_summary_id
    idf_weights: dict[str, float] = {}

    if bucketing_strategy == "graph":
        graph_input = graph_input or await load_validated_graph_input(
            bucketing_strategy,
            dataset_id,
            text_summaries,
        )
        entities_by_summary_id = graph_input.entities_by_summary_id
        idf_weights = graph_input.idf_weights

    context_datapoints, assignments = await build_context_index(
        new_text_summaries=new_summaries,
        text_summaries_all=text_summaries,
        buckets_by_level=group_buckets_by_level(context_input.buckets),
        existing_root=context_input.root,
        dataset_id=dataset_id,
        vector_engine=unified_engine.vector,
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        bucketing_strategy=bucketing_strategy,
        min_overlap=min_overlap,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
        ctx=ctx,
    )

    await persist_context_index_edges(assignments, unified_engine)

    return context_datapoints
