from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization.models import GlobalContextSummary

from .constants import GLOBAL_CONTEXT_SUMMARY_COLLECTION
from .ids import create_document_bucket_id, create_root_summary_id


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
    """Give the index write a dataset-level data item, so its nodes and edges are stamped.

    The index is built over the whole dataset, not from one data item, so the
    memify pipeline hands it none. The sentinel id below makes every write
    carry a source ref of this dataset, so a dataset deletion on a shared
    graph removes the nodes and edges with the dataset's other refs. A
    document deletion does not reach them this way: it removes the document's
    bucket and the root by id (``remove_document_index``).
    """
    if ctx is None or getattr(ctx.data_item, "id", None) is not None:
        return ctx

    dataset_id = getattr(ctx.dataset, "id", ctx.dataset)
    ctx.data_item = SimpleNamespace(
        id=uuid5(NAMESPACE_URL, f"cognee:global-context-index:{dataset_id}")
    )
    return ctx


def build_context_index_edges(
    pairs: list[tuple[str, str]],
    relationship_name: str,
) -> list:
    edges = []
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for source_id, target_id in pairs:
        edges.append(
            (
                source_id,
                target_id,
                relationship_name,
                {
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "relationship_name": relationship_name,
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


async def remove_document_index(dataset_id: UUID | str, data_ids: list[UUID | str]) -> None:
    """Drop the deleted documents' buckets and the dataset's root, by their derived ids.

    Explicit, not by provenance: the index nodes carry the dataset-level
    sentinel ref of the build (``ensure_global_context_storage_context``), and
    on a ledger graph they carry no ref at all, so a document's own delete
    never reaches them. The ids derive from the dataset and the ``Data`` rows,
    like the routing rows', so this works on every graph the same way. The root
    goes too: it is computed from every document, so it would keep naming
    entities and documents the dataset no longer has, and ownership cannot
    express that (shared output survives to its last owner). The next index
    build writes everything anew. Takes a list so a sweep that deletes many
    documents removes them in one graph call and one vector delete, with the
    root once. Runs in the dataset's database context, like the document
    delete it follows.
    """
    dataset_key = str(dataset_id)
    await delete_context_index_nodes(
        await get_unified_engine(),
        [str(create_document_bucket_id(dataset_key, str(data_id))) for data_id in data_ids]
        + [str(create_root_summary_id(dataset_key))],
    )


def _context_index_source_ref(ctx: PipelineContext | None) -> tuple[str | None, str | None]:
    """Dataset-level source ref + run id for the global-context-index edges.

    Reuses the same sentinel data id as the root (``ensure_global_context_storage_context``),
    so the edges are removed with the dataset's other refs when the dataset is
    deleted. They are not tied to a document: an edge goes with either of its
    endpoints, and a bucket's edges therefore leave with the bucket when its
    document is deleted.
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
    edges_by_relationship: dict[str, list[tuple[str, str]]],
    unified_engine: Any,
    ctx: PipelineContext | None = None,
) -> None:
    """Write every kind of index edge in one statement.

    Keyed by relationship name, each with its (source id, target id) pairs,
    because the caller may write more than one kind at a time, and the
    provenance-marker read below is a round trip worth paying once per run
    rather than once per kind.
    """
    index_edges = [
        edge
        for relationship_name, pairs in edges_by_relationship.items()
        for edge in build_context_index_edges(pairs, relationship_name)
    ]
    if not index_edges:
        return

    # Structural global context index edges are written directly by this task.
    # On a graph-provenance graph, fold a dataset-level source ref into the write so
    # the edges are deletable/rollbackable; on ledger graphs they are unstamped
    # as before (the source ref is ignored).
    source_ref_key = None
    pipeline_run_id = None
    if await stores_provenance_in_graph(unified_engine.graph):
        source_ref_key, pipeline_run_id = _context_index_source_ref(ctx)

    await unified_engine.graph.add_edges(
        index_edges,
        source_ref_key=source_ref_key,
        pipeline_run_id=pipeline_run_id,
    )
