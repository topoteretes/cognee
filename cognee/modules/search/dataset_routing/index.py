"""Keep the routing collection equal to the index rows search can address.

A dataset's routing rows are the global context index rows whose ids derive
from the relational database: its root (``create_root_summary_id``) and one
bucket per document (``create_document_bucket_id``, keyed by ``Data.id``).
``sync_dataset_routes`` runs after every index build and reconciles that set
against what the build computed: it writes the rows whose text is new or
changed and deletes the rest, so a deleted document, an emptied dataset
(``forget(memory_only=True)``) and a switch to ``method="llm"`` leave no
stale row. ``remove_dataset_routes`` covers the one case no build follows:
the dataset itself is deleted.

Nothing here runs with backend access control off. Every dataset then shares
one store, so there is nothing to route between, and the rows would land in
the store the datasets themselves use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from cognee.modules.data.methods.get_data_ids_by_dataset import get_data_ids_by_dataset
from cognee.shared.logging_utils import get_logger

from .store import DATASET_ROUTE_COLLECTION, DatasetRoute, get_dataset_route_engine

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


async def sync_dataset_routes(dataset_id: str, desired: list[GlobalContextSummary]) -> int:
    """Make the dataset's routing rows equal to ``desired``; returns how many rows changed.

    ``desired`` is the full index the build computed (changed or not), so a row
    missing from the collection, for instance after an upgrade that found the
    graph index already built, is written on the next build. A summary whose id
    the read path cannot derive (an ``llm``-method bucket) is not mirrored.
    """
    if not backend_access_control_enabled():
        return 0

    documents = await get_data_ids_by_dataset([UUID(dataset_id)])
    addressable = dataset_route_ids(dataset_id, documents.get(UUID(dataset_id), []))
    wanted = {
        str(summary.id): summary for summary in desired if str(summary.id) in set(addressable)
    }

    engine = get_dataset_route_engine()
    existing = {
        str(row.id): (row.payload or {}).get("text")
        for row in await engine.retrieve(DATASET_ROUTE_COLLECTION, addressable)
    }
    stale = [UUID(row_id) for row_id in existing if row_id not in wanted]
    changed = [
        DatasetRoute(
            id=summary.id, text=summary.text, dataset_id=dataset_id, is_root=summary.is_root
        )
        for row_id, summary in wanted.items()
        if existing.get(row_id) != summary.text
    ]
    if stale:
        await engine.delete_data_points(DATASET_ROUTE_COLLECTION, stale)
    if changed:
        await engine.create_data_points(DATASET_ROUTE_COLLECTION, changed)
    logger.info(
        "Dataset routing index for %s: %d row(s) written, %d removed",
        dataset_id,
        len(changed),
        len(stale),
    )
    return len(changed) + len(stale)


async def remove_dataset_routes(dataset_id: UUID | str, document_ids: list[UUID | str]) -> None:
    """Delete every routing row a dataset could have; for deleting the dataset itself."""
    if not backend_access_control_enabled():
        return
    await get_dataset_route_engine().delete_data_points(
        DATASET_ROUTE_COLLECTION,
        [UUID(row_id) for row_id in dataset_route_ids(dataset_id, document_ids)],
    )


async def prune_dataset_routes() -> None:
    """Delete the routing rows of every dataset in the registry; for ``prune_system``.

    Derived from the relational tables, so it must run before they are dropped.
    """
    if not backend_access_control_enabled():
        return
    row_ids = [
        UUID(row_id)
        for dataset_id, document_ids in (await get_data_ids_by_dataset(None)).items()
        for row_id in dataset_route_ids(dataset_id, document_ids)
    ]
    if row_ids:
        await get_dataset_route_engine().delete_data_points(DATASET_ROUTE_COLLECTION, row_ids)
