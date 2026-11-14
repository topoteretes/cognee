"""Keep the routing collections equal to the index rows search can address.

A dataset's routing rows are the global context index rows whose ids derive
from the relational database: its root (``create_root_summary_id``) and one
bucket per document (``create_document_bucket_id``, keyed by ``Data.id``).
``sync_dataset_routes`` runs after every index build and reconciles that set
against what the build computed in the dataset's own collection -- the one of
the model that built it (the dataset context guard has already refused a
build under a model that cannot serve the dataset, so the configured record
names it): it writes the rows whose text is new or changed and deletes the
rest. It can only address rows whose document still has a ``Data`` row, so
deletion removes rows itself, while their ids are still derivable:
``remove_dataset_routes`` with the deleted documents, which also takes the
root, since the root names every document and the index drops its own root at
the same time; ``prune_dataset_routes`` drops every routing collection, so a
prune starts from a clean store.

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
    dataset_route_collection,
    dataset_route_collection_name,
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

    The rows are not embedded again: ``index_engine`` is the dataset's own
    vector store, where the index build just embedded these summaries, and the
    rows it stored for them are copied as they are, vector and payload, so a
    routing row is the index row of every other ``<Type>_<field>`` collection
    (the adapter's ``IndexSchema``; ``upsert_raw_vectors`` with no payload
    schema stores that shape). One model per collection is what makes the copy
    correct: the build runs under the model the dataset's guard admitted, and
    that model's collection is where its rows belong. Rows of a model the
    dataset has moved away from keep the collection they were built under,
    where nothing scores them: the read path only addresses the ids of
    datasets the configured model can serve (``candidates.py``).
    """
    if not backend_access_control_enabled():
        return 0

    engine = await get_dataset_route_engine()
    collection = dataset_route_collection()

    documents = await get_data_ids_by_dataset([UUID(dataset_id)])
    addressable = set(dataset_route_ids(dataset_id, documents.get(UUID(dataset_id), [])))
    wanted = {str(summary.id): summary for summary in desired if str(summary.id) in addressable}

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
        try:
            await engine.upsert_raw_vectors(
                collection,
                [
                    {
                        "id": summary.id,
                        "vector": rows[str(summary.id)]["vector"],
                        "payload": {
                            key: value
                            for key, value in rows[str(summary.id)].items()
                            if key != "vector"
                        },
                    }
                    for summary in changed
                ],
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


async def _index_rows(
    index_engine, dataset_id: str, summaries: list[GlobalContextSummary]
) -> dict[str, dict]:
    """The rows the index build stored for ``summaries``, by id, from the dataset's store.

    Each is the stored payload with the vector under ``"vector"``, as
    ``retrieve(..., include_vector=True)`` returns it.
    """
    from cognee.tasks.memify.global_context_index.constants import (
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
    )

    rows = await index_engine.retrieve(
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
        [str(summary.id) for summary in summaries],
        include_vector=True,
    )
    stored = {str(row.id): row.payload for row in rows}
    missing = [str(summary.id) for summary in summaries if str(summary.id) not in stored]
    if missing:
        raise LookupError(
            f"Dataset routing: {len(missing)} index row(s) of dataset {dataset_id} have no "
            f"vector in {GLOBAL_CONTEXT_SUMMARY_COLLECTION}; the index is written before it "
            "is mirrored, so the routing rows copy its vectors instead of embedding again."
        )
    return stored


async def remove_dataset_routes(dataset_id: UUID | str, document_ids: list[UUID | str]) -> None:
    """Delete the dataset's root row and the bucket rows of the given documents, in every routing collection.

    Called with one document when it, or its memory, is deleted, and with all
    of them when the dataset is. The root goes either way: its text names every
    document, so the index drops it with any of them. After a document deletion
    the index writes a ``from_entities`` root again at once and mirrors it here
    (``rebuild_document_index_root``); the buckets wait for the next build.

    The rows may sit in any collection a build ever mirrored them into, so the
    deletion runs against all of them (``_all_route_collections``): deleting
    from a collection that does not exist is a no-op.
    """
    if not backend_access_control_enabled():
        return
    engine = await get_dataset_route_engine()
    row_ids = [UUID(row_id) for row_id in dataset_route_ids(dataset_id, document_ids)]
    for collection in await _all_route_collections(engine):
        await engine.delete_data_points(collection, row_ids)


async def prune_dataset_routes() -> None:
    """Drop every routing collection; for ``prune_system``.

    A collection's vector width is fixed when it is created, so emptying it
    would leave a prune followed by an ``EMBEDDING_MODEL`` change unable to
    write; dropping every collection lets the next index build create the
    configured model's at the new width.
    """
    if not backend_access_control_enabled():
        return
    engine = await get_dataset_route_engine()
    warned = False
    for collection in await _all_route_collections(engine):
        try:
            await engine.delete_collection(collection)
        except NotImplementedError:
            if not warned:
                logger.warning(
                    "The %s vector adapter cannot drop a collection; the dataset routing "
                    "collections were left in place.",
                    type(engine).__name__,
                )
                warned = True


async def _all_route_collections(engine) -> list[str]:
    """Every collection routing rows could live in, the ones that exist.

    The configured model's collection, every collection the registry's rows
    name (a dataset carries the model that built it, so the distinct records
    are the models this deployment has ever built with), and the pre-per-model
    collection an upgraded deployment still holds. A dataset whose rows were
    mirrored before its model left the registry has none left: its ids die
    with its ``Data`` rows, so no read can ever reach those rows again. A
    registry that cannot be read (a prune before any database was created,
    a store that is down) leaves the known names: the rows a deletion would
    miss there are unreachable through routing anyway.
    """
    names = [DATASET_ROUTE_COLLECTION, dataset_route_collection()]
    try:
        names += await _registry_route_collections()
    except Exception:
        logger.debug(
            "The recorded embedding models could not be read; cleaning the routing "
            "collections the configuration names.",
            exc_info=True,
        )
    deduped = list(dict.fromkeys(names))
    try:
        return await engine.existing_collections(deduped)
    except Exception:
        logger.debug(
            "Routing collections could not be listed; deleting in all of them.", exc_info=True
        )
        return deduped


async def _registry_route_collections() -> list[str]:
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        DIMENSIONS_KEY,
        MODEL_KEY,
    )
    from cognee.modules.users.models import DatasetDatabase

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (await session.scalars(select(DatasetDatabase))).all()

    names = []
    for row in rows:
        info = row.vector_database_connection_info or {}
        model = info.get(MODEL_KEY)
        if model is None:
            # A legacy row records a width only; no collection was named for it.
            continue
        name = dataset_route_collection_name(model, info.get(DIMENSIONS_KEY))
        if name not in names:
            names.append(name)
    return names
