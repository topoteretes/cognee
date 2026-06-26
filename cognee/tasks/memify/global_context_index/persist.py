from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import is_graph_native_graph
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization.models import GlobalContextSummary

from .constants import GLOBAL_CONTEXT_SUMMARY_COLLECTION, SUMMARIZED_IN
from .models import BucketAssignment


async def safe_delete_context_vectors(vector_engine: Any, ids: list[str]) -> None:
    if not ids:
        return

    try:
        await vector_engine.delete_data_points(
            GLOBAL_CONTEXT_SUMMARY_COLLECTION,
            [UUID(str(item_id)) for item_id in ids],
        )
    except (CollectionNotFoundError, ValueError):
        return


async def delete_context_index_nodes(unified_engine: Any, bucket_ids: list[str]) -> None:
    if not bucket_ids:
        return
    await unified_engine.graph.delete_nodes(bucket_ids)
    await safe_delete_context_vectors(unified_engine.vector, bucket_ids)


def ensure_global_context_storage_context(
    ctx: PipelineContext | None,
) -> PipelineContext | None:
    if ctx is None or getattr(ctx.data_item, "id", None) is not None:
        return ctx

    dataset_id = getattr(ctx.dataset, "id", ctx.dataset)
    ctx.data_item = SimpleNamespace(
        id=uuid5(NAMESPACE_URL, f"cognee:global-context-index:{dataset_id}")
    )
    return ctx


def build_context_index_edges(assignments: list[BucketAssignment]) -> list:
    edges = []
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for assignment in assignments:
        edges.append(
            (
                assignment.child_id,
                assignment.parent_id,
                SUMMARIZED_IN,
                {
                    "source_node_id": assignment.child_id,
                    "target_node_id": assignment.parent_id,
                    "relationship_name": SUMMARIZED_IN,
                    "updated_at": updated_at,
                },
            )
        )

    return edges


async def persist_context_summaries(
    summary_datapoints: list[GlobalContextSummary],
    ctx: PipelineContext | None,
) -> None:
    if summary_datapoints:
        await add_data_points(
            summary_datapoints,
            ctx=ensure_global_context_storage_context(ctx),
        )


def _context_index_source_ref(ctx: PipelineContext | None) -> tuple[str | None, str | None]:
    """Dataset-level source ref + run id for the global-context-index edges.

    Reuses the same sentinel data id as the context summary nodes
    (``ensure_global_context_storage_context``) so the ``summarized_in`` edges
    carry the same provenance key as the summaries they connect — both are then
    removed together when the dataset is deleted. These edges are dataset-scoped
    (built over many summaries), so they are deliberately not tied to a single
    ingested data item.
    """
    ctx = ensure_global_context_storage_context(ctx)
    if ctx is None:
        return None, None
    dataset_id = getattr(ctx.dataset, "id", ctx.dataset)
    data_id = getattr(ctx.data_item, "id", None)
    if dataset_id is None or data_id is None:
        return None, None
    pipeline_run_id = getattr(ctx, "pipeline_run_id", None)
    return (
        make_source_ref_key(dataset_id, data_id),
        str(pipeline_run_id) if pipeline_run_id else None,
    )


async def persist_context_index_edges(
    assignments: list[BucketAssignment],
    unified_engine: Any,
    ctx: PipelineContext | None = None,
) -> None:
    summarized_in_edges = build_context_index_edges(assignments)
    if not summarized_in_edges:
        return

    # Structural global context index edges are written directly by this task.
    # On a graph-native graph, fold a dataset-level source ref into the write so
    # the edges are deletable/rollbackable; on ledger graphs they are unstamped
    # as before (the source ref is ignored).
    source_ref_key = None
    pipeline_run_id = None
    if await is_graph_native_graph(unified_engine.graph):
        source_ref_key, pipeline_run_id = _context_index_source_ref(ctx)

    await unified_engine.graph.add_edges(
        summarized_in_edges,
        source_ref_key=source_ref_key,
        pipeline_run_id=pipeline_run_id,
    )
