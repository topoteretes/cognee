"""Keep the routing collection equal to the index rows search can address.

A dataset's routing rows are the global context index rows whose ids derive
from the relational database: its root (``create_root_summary_id``) and one
bucket per document (``create_document_bucket_id``, keyed by ``Data.id``).
``sync_dataset_routes`` runs after every index build and reconciles that set
against what the build computed: it writes the rows whose text is new or
changed and deletes the rest. It can only address rows whose document still
has a ``Data`` row, so deletion removes rows itself, while their ids are still
derivable: ``remove_dataset_routes`` with the deleted documents, which also
takes the root, since the root names every document and the index drops its
own root at the same time. Removal covers every routing collection;
``prune_dataset_routes`` drops all named and legacy routing collections.

Nothing here runs with backend access control off. Every dataset then shares
one store, so there is nothing to route between, and the rows would land in
the store the datasets themselves use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import (
    embedding_request_scope,
)
from cognee.modules.data.methods.get_data_ids_by_dataset import get_data_ids_by_dataset
from cognee.shared.logging_utils import get_logger

from .store import (
    dataset_route_collections,
    get_dataset_route_embedding,
    get_dataset_route_engine,
)

if TYPE_CHECKING:
    from cognee.tasks.summarization.models import GlobalContextSummary

logger = get_logger("dataset_routing")


def backend_access_control_enabled() -> bool:
    # Imported at call time: this module is reached from the data methods
    # package, which the context module's own imports lead back to.
    from cognee.context_global_variables import backend_access_control_enabled

    return backend_access_control_enabled()


def dataset_route_ids(dataset_id: UUID | str, document_ids: list[UUID | str]) -> list[str]:
    """The routing row ids a dataset can have: its root, then one bucket per document."""
    # The id functions live with the index that writes the nodes; imported here
    # because that package imports the task that calls back into this module.
    from cognee.tasks.memify.global_context_index.ids import (
        create_document_bucket_id,
        create_root_summary_id,
    )

    dataset_key = str(dataset_id)
    return [str(create_root_summary_id(dataset_key))] + [
        str(create_document_bucket_id(dataset_key, str(document_id)))
        for document_id in document_ids
    ]


async def sync_dataset_routes(
    dataset_id: str, desired: list[GlobalContextSummary], index_engine
) -> int:
    """Make the dataset's routing rows equal to ``desired``; returns how many rows changed.

    ``desired`` is the full index the build computed (changed or not), so a row
    missing from the collection, for instance after an upgrade that found the
    graph index already built, is written on the next build. A summary whose id
    the read path cannot derive (an ``llm``-method bucket) is not mirrored.
    Source payloads retain their index shape and tags. New or changed text is
    embedded together in deployment space; source vectors are never read or
    modified. Noncurrent deployment collections are invalidated first so an
    older model cannot score stale summaries after a later switch.
    """
    if not backend_access_control_enabled():
        return 0

    engine = await get_dataset_route_engine()
    embedding = await get_dataset_route_embedding()
    collection = embedding.collection

    documents = await get_data_ids_by_dataset([UUID(dataset_id)])
    addressable = set(dataset_route_ids(dataset_id, documents.get(UUID(dataset_id), [])))
    wanted = {str(summary.id): summary for summary in desired if str(summary.id) in addressable}

    collections = await dataset_route_collections(engine)
    addressable_ids = [UUID(row_id) for row_id in addressable]
    for old_collection in collections:
        if old_collection.lower() != collection.lower():
            await engine.delete_data_points(old_collection, addressable_ids)

    existing = {
        str(row.id): (row.payload or {}).get("text")
        for row in await engine.retrieve(collection, list(addressable))
    }
    stale = [UUID(row_id) for row_id in existing if row_id not in wanted]
    changed = [
        summary for row_id, summary in wanted.items() if existing.get(row_id) != summary.text
    ]
    if stale:
        await engine.delete_data_points(collection, stale)
    if changed:
        rows = await _index_rows(index_engine, dataset_id, changed)
        with embedding_request_scope(embedding.requests):
            vectors = await embedding.engine.embed_text(
                [rows[str(summary.id)]["text"] for summary in changed]
            )
        if len(vectors) != len(changed):
            raise ValueError("Dataset routing embedder returned an incorrect number of vectors")
        await engine.upsert_raw_vectors(
            collection,
            [
                {"id": summary.id, "vector": vector, "payload": rows[str(summary.id)]}
                for summary, vector in zip(changed, vectors)
            ],
            vector_size=embedding.vector_size,
        )
    logger.info(
        "Dataset routing index for %s: %d row(s) written, %d removed",
        dataset_id,
        len(changed),
        len(stale),
    )
    return len(changed) + len(stale)


async def _index_rows(
    index_engine, dataset_id: str, summaries: list[GlobalContextSummary]
) -> dict[str, dict]:
    """The rows the index build stored for ``summaries``, by id, from the dataset's store.

    Payload-only retrieval leaves the dataset's vectors in its own space.
    """
    from cognee.tasks.memify.global_context_index.constants import (
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
    )

    rows = await index_engine.retrieve(
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
        [str(summary.id) for summary in summaries],
    )
    stored = {str(row.id): row.payload for row in rows}
    missing = [str(summary.id) for summary in summaries if str(summary.id) not in stored]
    if missing:
        raise LookupError(
            f"Dataset routing: {len(missing)} index row(s) of dataset {dataset_id} have no "
            f"payload in {GLOBAL_CONTEXT_SUMMARY_COLLECTION}; the index must be written "
            "before its routing summaries."
        )
    return stored


async def remove_dataset_routes(dataset_id: UUID | str, document_ids: list[UUID | str]) -> None:
    """Delete the dataset's root row and the bucket rows of the given documents.

    Called with one document when it, or its memory, is deleted, and with all
    of them when the dataset is. The root goes either way: its text names every
    document, so the index drops it with any of them. After a document deletion
    the index writes a ``from_entities`` root again at once and mirrors it here
    (``rebuild_document_index_root``); the buckets wait for the next build.
    """
    if not backend_access_control_enabled():
        return
    engine = await get_dataset_route_engine()
    ids = [UUID(row_id) for row_id in dataset_route_ids(dataset_id, document_ids)]
    for collection in await dataset_route_collections(engine):
        await engine.delete_data_points(collection, ids)


async def prune_dataset_routes() -> None:
    """Drop every routing collection, including orphaned/legacy ones, for ``prune_system``."""
    if not backend_access_control_enabled():
        return
    engine = await get_dataset_route_engine()
    for collection in await dataset_route_collections(engine):
        await engine.delete_collection(collection)
