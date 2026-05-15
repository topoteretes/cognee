from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

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


def build_context_index_edges(assignments: list[BucketAssignment]) -> list:
    edges = []
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for assignment in assignments:
        edges.append(
            (
                assignment.summary_id,
                assignment.bucket_id,
                SUMMARIZED_IN,
                {
                    "source_node_id": assignment.summary_id,
                    "target_node_id": assignment.bucket_id,
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
        await add_data_points(summary_datapoints, ctx=ctx)


async def persist_context_index_edges(
    assignments: list[BucketAssignment],
    unified_engine: Any,
) -> None:
    summarized_in_edges = build_context_index_edges(assignments)
    if summarized_in_edges:
        # Structural global context index edges are written directly by this task.
        await unified_engine.graph.add_edges(summarized_in_edges)
