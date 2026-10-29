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
own root at the same time; ``prune_dataset_routes`` drops the whole
collection, so a prune followed by an embedding model change starts from a
clean table.

Nothing here runs with backend access control off. Every dataset then shares
one store, so there is nothing to route between, and the rows would land in
the store the datasets themselves use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from cognee.modules.data.methods.get_data_ids_by_dataset import get_data_ids_by_dataset
from cognee.shared.logging_utils import get_logger

from .store import (
    DATASET_ROUTE_COLLECTION,
    DatasetRoute,
    get_dataset_route_engine,
    routing_width_matches,
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
    Nothing is written while the collection's vector width differs from the
    configured model's; the rows wait for a prune.

    The rows are not embedded again: ``index_engine`` is the dataset's own
    vector store, where the index build just embedded these summaries, and
    their vectors are read back from it and written as they are. One embedding
    model for the deployment is what makes the copy correct, and it is the
    assumption routing rests on anyway (the width check above is its guard).
    """
    if not backend_access_control_enabled():
        return 0

    engine = await get_dataset_route_engine()
    if not await routing_width_matches(engine):
        return 0

    documents = await get_data_ids_by_dataset([UUID(dataset_id)])
    addressable = set(dataset_route_ids(dataset_id, documents.get(UUID(dataset_id), [])))
    wanted = {str(summary.id): summary for summary in desired if str(summary.id) in addressable}

    existing = {
        str(row.id): (row.payload or {}).get("text")
        for row in await engine.retrieve(DATASET_ROUTE_COLLECTION, list(addressable))
    }
    stale = [UUID(row_id) for row_id in existing if row_id not in wanted]
    changed = [
        summary for row_id, summary in wanted.items() if existing.get(row_id) != summary.text
    ]
    if stale:
        await engine.delete_data_points(DATASET_ROUTE_COLLECTION, stale)
    if changed:
        vectors = await _index_vectors(index_engine, dataset_id, changed)
        try:
            await engine.upsert_raw_vectors(
                DATASET_ROUTE_COLLECTION,
                [
                    {
                        "id": summary.id,
                        "vector": vectors[str(summary.id)],
                        "payload": DatasetRoute(
                            id=summary.id,
                            text=summary.text,
                            dataset_id=dataset_id,
                            is_root=summary.is_root,
                        ).model_dump(),
                    }
                    for summary in changed
                ],
                payload_schema=DatasetRoute,
            )
        except NotImplementedError:
            logger.warning(
                "The %s vector adapter cannot write precomputed vectors; the dataset routing "
                "rows of %s were not written and the dataset is searched without routing.",
                type(engine).__name__,
                dataset_id,
            )
            return len(stale)
    logger.info(
        "Dataset routing index for %s: %d row(s) written, %d removed",
        dataset_id,
        len(changed),
        len(stale),
    )
    return len(changed) + len(stale)


async def _index_vectors(
    index_engine, dataset_id: str, summaries: list[GlobalContextSummary]
) -> dict[str, list[float]]:
    """The vectors the index build stored for ``summaries``, by id, from the dataset's store."""
    from cognee.tasks.memify.global_context_index.constants import (
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
    )

    rows = await index_engine.retrieve(
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
        [str(summary.id) for summary in summaries],
        include_vector=True,
    )
    vectors = {str(row.id): row.payload["vector"] for row in rows}
    missing = [str(summary.id) for summary in summaries if str(summary.id) not in vectors]
    if missing:
        raise LookupError(
            f"Dataset routing: {len(missing)} index row(s) of dataset {dataset_id} have no "
            f"vector in {GLOBAL_CONTEXT_SUMMARY_COLLECTION}; the index is written before it "
            "is mirrored, so the routing rows copy its vectors instead of embedding again."
        )
    return vectors


async def remove_dataset_routes(dataset_id: UUID | str, document_ids: list[UUID | str]) -> None:
    """Delete the dataset's root row and the bucket rows of the given documents.

    Called with one document when it, or its memory, is deleted, and with all
    of them when the dataset is. The root goes either way: its text names every
    document, so the index drops it with any of them, and the next build
    writes both anew.
    """
    if not backend_access_control_enabled():
        return
    await (await get_dataset_route_engine()).delete_data_points(
        DATASET_ROUTE_COLLECTION,
        [UUID(row_id) for row_id in dataset_route_ids(dataset_id, document_ids)],
    )


async def prune_dataset_routes() -> None:
    """Drop the routing collection; for ``prune_system``.

    The collection's vector width is fixed when it is created, so emptying it
    would leave a prune followed by an ``EMBEDDING_MODEL`` change unable to
    write; dropping it lets the next index build create it at the new width.
    """
    if not backend_access_control_enabled():
        return
    engine = await get_dataset_route_engine()
    try:
        await engine.delete_collection(DATASET_ROUTE_COLLECTION)
    except NotImplementedError:
        logger.warning(
            "The %s vector adapter cannot drop a collection; the dataset routing "
            "collection %s was left in place.",
            type(engine).__name__,
            DATASET_ROUTE_COLLECTION,
        )
