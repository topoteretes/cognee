from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.provenance import make_source_ref_key, parse_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization.models import GlobalContextSummary

from .constants import GLOBAL_CONTEXT_SUMMARY_COLLECTION
from .ids import create_root_summary_id


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
    carry a source ref of this dataset: the root and the edges keep it, so a
    dataset deletion on a shared graph removes them with the dataset's other
    refs, and a bucket trades it for its document's own ref right after the
    write (``stamp_bucket_owners``).
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


async def stamp_bucket_owners(
    graph_engine: Any, dataset_id: str, buckets: list[GlobalContextSummary]
) -> None:
    """Give every document bucket its document's source ref, and only that one.

    A bucket is what the index says about one document, so the document owns
    it: ``forget()`` then removes the bucket through the provenance delete,
    like every other node the document produced, and nothing has to know the
    bucket's id at deletion time. The write folds the dataset-level ref in
    (``ensure_global_context_storage_context``), so a bucket is never stored
    unstamped; this pass replaces that ref with the document's. It runs over
    the whole index, not only the buckets just written, so buckets written
    before it existed, or by a build that crashed before reaching it, are
    stamped by the next build. Graph-provenance graphs only: a ledger graph
    keeps a document's bucket until the next build no longer computes it.
    """
    if not buckets or not await stores_provenance_in_graph(graph_engine):
        return

    dataset_uuid = UUID(dataset_id)
    current = await graph_engine.get_node_delete_data([str(bucket.id) for bucket in buckets])
    to_attach: dict[str, list[str]] = {}
    to_remove: dict[tuple[str, ...], list[str]] = {}
    for bucket in buckets:
        owner = make_source_ref_key(dataset_uuid, UUID(bucket.document_id))
        existing = current.get(str(bucket.id))
        held = set(existing.source_ref_keys if existing else ())
        if owner not in held:
            to_attach.setdefault(owner, []).append(str(bucket.id))
        stale = tuple(
            sorted(ref for ref in held - {owner} if _is_document_ref_of(ref, dataset_uuid))
        )
        if stale:
            to_remove.setdefault(stale, []).append(str(bucket.id))
    # One call per distinct key set: every bucket has its own owner, so the
    # attach is per bucket, but the refs to take back are shared and go in one.
    for owner, node_ids in to_attach.items():
        await graph_engine.attach_node_source_refs(node_ids, [owner])
    for stale, node_ids in to_remove.items():
        await graph_engine.remove_node_source_refs(node_ids, list(stale))


def _is_document_ref_of(source_ref_key: str, dataset_id: UUID) -> bool:
    """Whether a ref is a document-level ref of this dataset, the kind this pass owns."""
    parsed = parse_source_ref_key(source_ref_key)
    return parsed.dataset_id == dataset_id and parsed.chunk_id is None


async def remove_dataset_root(dataset_id: UUID | str) -> None:
    """Drop the dataset's root summary: a document it names was deleted.

    The root is computed from every document, so a document's deletion leaves
    it naming entities and a document the dataset no longer has. Ownership
    cannot express that (shared output survives to its last owner), so the
    root goes with the document, and the next index build writes a new one.
    Runs in the dataset's database context, like the document delete it follows.
    """
    await delete_context_index_nodes(
        await get_unified_engine(), [str(create_root_summary_id(str(dataset_id)))]
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
