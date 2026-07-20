from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.graph.methods.get_global_context_graph_inputs import (
    get_dataset_text_summary_ids,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import GlobalContextSummary

from .bucketing.graph.inputs import load_graph_bucketing_inputs
from .bucketing.graph.placement import (
    validate_graph_buckets_can_be_extended,
    validate_vector_buckets_can_be_extended,
)
from .build import build_context_index, group_buckets_by_level
from .bucketing_strategy import BucketingStrategyName
from .load import dataset_id_from_context, load_context_index_input_from_graph
from .models import GlobalContextIndexUpdateData, SummaryNode
from .persist import delete_context_index_nodes, persist_context_index_edges

logger = get_logger("global_context_index")
GraphBucketingInputs = tuple[dict[str, set[str]], dict[str, float]]


@dataclass
class UpdateScope:
    dataset_id: str
    context_input: GlobalContextIndexUpdateData
    text_summaries: list[SummaryNode]


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


def validate_graph_dataset_context(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
) -> None:
    if bucketing_strategy == "graph" and not dataset_id:
        raise ValueError('bucketing_strategy="graph" requires a dataset context.')


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


async def select_text_summaries_for_strategy(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
    text_summaries: list[SummaryNode],
) -> list[SummaryNode]:
    text_summaries = [summary for summary in text_summaries if summary.text]
    if bucketing_strategy != "graph":
        return text_summaries
    return await filter_graph_dataset_text_summaries(dataset_id, text_summaries)


async def load_graph_bucketing_data_if_needed(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
    text_summaries: list[SummaryNode],
) -> GraphBucketingInputs | None:
    if bucketing_strategy != "graph" or not text_summaries:
        return None

    return await load_graph_bucketing_inputs(
        dataset_id,
        [summary.id for summary in text_summaries],
    )


async def reset_context_index_for_rebuild(
    context_input: GlobalContextIndexUpdateData,
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


def validate_existing_buckets_for_strategy(
    bucketing_strategy: BucketingStrategyName,
    existing_buckets: list[SummaryNode],
) -> None:
    if bucketing_strategy == "graph":
        validate_graph_buckets_can_be_extended(existing_buckets)
        return

    validate_vector_buckets_can_be_extended(existing_buckets)


async def load_update_scope(
    data: Any,
    bucketing_strategy: BucketingStrategyName,
    ctx: PipelineContext | None,
) -> UpdateScope:
    dataset_id = dataset_id_from_context(ctx)
    validate_graph_dataset_context(bucketing_strategy, dataset_id)

    context_input = data
    if context_input is None or context_input == {}:
        context_input = await load_context_index_input_from_graph(ctx)
    if not isinstance(context_input, GlobalContextIndexUpdateData):
        raise TypeError(
            "update_global_context_index expected GlobalContextIndexUpdateData, None, or {}."
        )

    text_summaries = await select_text_summaries_for_strategy(
        bucketing_strategy,
        dataset_id,
        context_input.text_summaries,
    )
    return UpdateScope(dataset_id, context_input, text_summaries)


async def prepare_existing_context_index(
    scope: UpdateScope,
    rebuild: bool,
    bucketing_strategy: BucketingStrategyName,
) -> tuple[Any | None, GraphBucketingInputs | None]:
    if not rebuild:
        validate_existing_buckets_for_strategy(
            bucketing_strategy,
            scope.context_input.buckets,
        )
        return None, None

    graph_bucketing_inputs = await load_graph_bucketing_data_if_needed(
        bucketing_strategy,
        scope.dataset_id,
        scope.text_summaries,
    )
    unified_engine = await get_unified_engine()
    await reset_context_index_for_rebuild(
        scope.context_input,
        scope.text_summaries,
        unified_engine,
    )
    return unified_engine, graph_bucketing_inputs


async def ensure_graph_bucketing_inputs(
    bucketing_strategy: BucketingStrategyName,
    dataset_id: str,
    text_summaries: list[SummaryNode],
    graph_bucketing_inputs: GraphBucketingInputs | None,
) -> GraphBucketingInputs | None:
    if bucketing_strategy != "graph":
        return None
    if graph_bucketing_inputs is not None:
        return graph_bucketing_inputs
    return await load_graph_bucketing_data_if_needed(
        bucketing_strategy,
        dataset_id,
        text_summaries,
    )


def unpack_graph_bucketing_inputs(
    graph_bucketing_inputs: GraphBucketingInputs | None,
) -> tuple[dict[str, set[str]], dict[str, float]]:
    if graph_bucketing_inputs is None:
        return {}, {}
    return graph_bucketing_inputs


async def build_and_persist_context_index(
    scope: UpdateScope,
    new_summaries: list[SummaryNode],
    unified_engine: Any,
    max_bucket_size: int,
    placement_distance_threshold: float,
    bucketing_strategy: BucketingStrategyName,
    min_overlap: float,
    graph_bucketing_inputs: GraphBucketingInputs | None,
    ctx: PipelineContext | None,
) -> list[GlobalContextSummary]:
    entities_by_summary_id, idf_weights = unpack_graph_bucketing_inputs(graph_bucketing_inputs)
    context_datapoints, assignments = await build_context_index(
        new_text_summaries=new_summaries,
        text_summaries_all=scope.text_summaries,
        buckets_by_level=group_buckets_by_level(scope.context_input.buckets),
        existing_root=scope.context_input.root,
        dataset_id=scope.dataset_id,
        vector_engine=unified_engine.vector,
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        bucketing_strategy=bucketing_strategy,
        min_overlap=min_overlap,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
        ctx=ctx,
    )

    await persist_context_index_edges(assignments, unified_engine, ctx=ctx)
    return context_datapoints


@task_summary("Updated {n} global context index node(s)")
async def update_global_context_index(
    data: Any = None,
    rebuild: bool = False,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
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
    scope = await load_update_scope(data, bucketing_strategy, ctx)
    unified_engine, graph_bucketing_inputs = await prepare_existing_context_index(
        scope,
        rebuild,
        bucketing_strategy,
    )

    if not scope.text_summaries:
        return []

    new_summaries = select_new_context_index_items(scope.text_summaries, rebuild)
    if not new_summaries:
        logger.info("No new TextSummary nodes to place in global context index.")
        return []

    graph_bucketing_inputs = await ensure_graph_bucketing_inputs(
        bucketing_strategy,
        scope.dataset_id,
        scope.text_summaries,
        graph_bucketing_inputs,
    )
    unified_engine = unified_engine or await get_unified_engine()

    return await build_and_persist_context_index(
        scope,
        new_summaries,
        unified_engine,
        max_bucket_size,
        placement_distance_threshold,
        bucketing_strategy,
        min_overlap,
        graph_bucketing_inputs,
        ctx,
    )
