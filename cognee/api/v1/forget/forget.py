from enum import Enum
from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.observability import (
    COGNEE_DATASET_NAME,
    COGNEE_FORGET_TARGET,
    COGNEE_RESULT_COUNT,
    MEMORY_OPERATION,
    MEMORY_SYSTEM,
    increment_items_deleted,
    new_span,
    record_operation_duration,
)
from cognee.modules.operations import record_operation
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME

logger = get_logger("forget")


class ForgetTarget(str, Enum):
    """What forget() reports it removed, picked from its arguments (telemetry and tracing)."""

    EVERYTHING = "everything"
    DATA_ITEM_MEMORY_ONLY = "data_item_memory_only"
    DATASET_MEMORY_ONLY = "dataset_memory_only"
    DATA_ITEM = "data_item"
    DATASET = "dataset"
    SEARCH_HISTORY = "search_history"
    UNKNOWN = "unknown"


async def forget(
    *,
    data_id: UUID | None = None,
    dataset: str | None = None,
    dataset_id: UUID | None = None,
    everything: bool = False,
    memory_only: bool = False,
    clean_history: bool = False,
    user: Any = None,
) -> dict:
    """Remove data from the knowledge graph.

    Unified deletion command that replaces the separate prune/delete/
    empty_dataset APIs with a single mental model.

    Usage patterns::

        # Forget a specific data item from a dataset
        await cognee.forget(data_id=data_id, dataset_id=dataset_id)

        # Forget an entire dataset (all data + graph nodes + vector entries)
        await cognee.forget(dataset="scientists")

        # Forget everything the current user owns
        await cognee.forget(everything=True)

        # Forget only memory (graph + vector) for a dataset (keep raw files)
        await cognee.forget(dataset="scientists", memory_only=True)

        # Forget only memory for a single file in a dataset
        await cognee.forget(dataset="scientists", data_id=data_id, memory_only=True)

        # Forget one item and the caller's own search history with it
        await cognee.forget(data_id=data_id, dataset_id=dataset_id, clean_history=True)

        # Delete only the caller's own search history, no data
        await cognee.forget(clean_history=True)

    Args:
        data_id: UUID of a specific data item to remove.
            Requires ``dataset`` or ``dataset_id`` to also be set. Needs
            ``delete`` on the dataset, or ``write`` when the caller added the
            item themselves (``memory_only`` always needs ``delete``).
        dataset: Dataset name. When set alone, deletes the
            entire dataset. When set with ``data_id``, deletes that item
            from this dataset.
        dataset_id: Dataset UUID. Alternative to ``dataset``.
        everything: If True, delete every dataset the user may delete, and every
            session the user owns or that belongs to one of those datasets.
            Other users' sessions stay: the session cache is shared and is never
            pruned here. Ignores ``data_id``, ``dataset``, and ``dataset_id``.
        memory_only: If True (requires ``dataset`` or ``dataset_id``), delete only memory
            (graph nodes/edges and vector embeddings) and reset pipeline status.
            Raw files and data records are preserved so the dataset can
            be re-cognified with different settings. A search that finds nothing
            afterwards does not mean the data is gone: the document is still
            stored and the next cognify processes it again.
        clean_history: If True, also delete the caller's own search and recall
            history: every ``queries``/``results`` row this user owns, in any
            dataset, the same rows ``DELETE /api/v1/search/history`` removes.
            It runs before the data is deleted and raises if it fails, so the
            forget can be retried. Default False: a document forget keeps
            history, because answers are recorded per dataset, not per
            document. A dataset or everything forget purges the deleted
            datasets' history for every user regardless of this flag. Alone,
            with no target, the call deletes only the history.
        user: User context. Resolved to default user when None.

    Returns:
        Dict with deletion summary: items removed, datasets removed, and
        ``history_cleaned`` (whether the caller's own search history was removed).

    What stays on disk:
        Every mode removes the records and the memory it names; it does not
        scrub the bytes from the storage engines. LanceDB keeps deleted rows in
        its append-only fragments until compaction rewrites them, and it
        rewrites a fragment only once enough of it is deleted, so a small
        delete in a large table can stay on disk indefinitely. The embedded
        graph file has no user-facing compaction. A byte-level erasure
        guarantee needs the engine's own compaction or vacuum, or a new store.
    """
    from cognee import __version__ as cognee_version
    from cognee.shared.utils import send_telemetry

    dataset_ref = dataset_id or dataset

    if dataset and dataset_id:
        raise ValueError("Provide either dataset or dataset_id, not both.")

    if everything:
        target = ForgetTarget.EVERYTHING
    elif memory_only and data_id:
        target = ForgetTarget.DATA_ITEM_MEMORY_ONLY
    elif memory_only and dataset_ref:
        target = ForgetTarget.DATASET_MEMORY_ONLY
    elif data_id:
        target = ForgetTarget.DATA_ITEM
    elif dataset_ref:
        target = ForgetTarget.DATASET
    elif clean_history:
        target = ForgetTarget.SEARCH_HISTORY
    else:
        target = ForgetTarget.UNKNOWN

    send_telemetry(
        "cognee.forget",
        user if user and hasattr(user, "id") else "sdk",
        additional_properties={
            "target": target.value,
            "dataset_name": dataset or "",
            "dataset_id": str(dataset_id) if dataset_id else "",
            "data_id": str(data_id) if data_id else "",
            "cognee_version": cognee_version,
        },
    )

    import time as _time

    _forget_start_ns = _time.monotonic_ns()

    with new_span("memory.delete") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "delete")
        span.set_attribute(COGNEE_FORGET_TARGET, target.value)
        if dataset_ref:
            span.set_attribute(COGNEE_DATASET_NAME, str(dataset_ref))

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            result = await client.forget(
                data_id=data_id,
                dataset=dataset,
                dataset_id=dataset_id,
                everything=everything,
                memory_only=memory_only,
                clean_history=clean_history,
            )
            span.set_attribute(
                COGNEE_RESULT_COUNT,
                result.get("datasets_removed", 0) if isinstance(result, dict) else 0,
            )
            return result

        async with record_operation("forget", user=user) as operation_context:
            # In case there is no database, forget will fail when getting a user
            from cognee.low_level import setup
            from cognee.modules.users.methods import get_default_user

            await setup()

            if user is None:
                user = await get_default_user()
            operation_context.set_user(user)

            # Validate the target before any side effect.
            if dataset_ref is None and not everything:
                if memory_only:
                    raise ValueError("memory_only requires dataset or dataset_id.")
                if data_id is not None:
                    raise ValueError("data_id requires dataset or dataset_id.")
                if not clean_history:
                    raise ValueError(
                        "Specify dataset, dataset_id, data_id+dataset, everything=True, "
                        "or clean_history=True."
                    )

            # Before any data is deleted and not caught, like the dataset purges:
            # if it fails nothing is gone yet and the forget can be retried.
            # `everything` removes the caller's history itself (_forget_everything).
            if clean_history and not everything:
                from cognee.modules.search.operations import delete_user_search_history

                removed = await delete_user_search_history(user.id)
                logger.info(
                    "forget: cleaned search history for user=%s (%d queries, %d results)",
                    user.id,
                    removed["queries_deleted"],
                    removed["results_deleted"],
                )
                if dataset_ref is None:
                    # clean_history alone: the history was the whole request.
                    return {"history_cleaned": True, "status": "success"}

            # `everything` deletes across all datasets; the per-dataset DB context is
            # established per-dataset inside datasets.delete_all -> empty_dataset, so we
            # must NOT enter a single-dataset context here (dataset_ref is None, which
            # would try to create a dataset_database row for a non-existent dataset).
            if everything:
                result = await _forget_everything(user)
                _removed = result.get("datasets_removed", 0)
                span.set_attribute(COGNEE_RESULT_COUNT, _removed)
                _duration_ms = (_time.monotonic_ns() - _forget_start_ns) / 1_000_000
                _attrs = {"memory.system": "cognee", "memory.operation": "delete"}
                record_operation_duration(_duration_ms, _attrs)
                increment_items_deleted(_removed, _attrs)
                return result

            # All remaining operations are scoped to a single dataset.
            # Authorize before any dataset database context is entered: context
            # entry provisions per-dataset database registry rows, a write that
            # must never happen for a caller without delete permission (an
            # unauthorized caller used to surface as a UniqueViolation 500 — or,
            # for an unprovisioned dataset, actually created rows). Removing one
            # item outright also admits its author with write permission;
            # memory_only stays delete-only (update() rebuilds through it).
            resolved_dataset_id = await _resolve_dataset_id(
                dataset_ref, user, data_id=None if memory_only else data_id
            )
            operation_context.set_dataset(resolved_dataset_id)

            # No database context here: each path below establishes it AFTER
            # acquiring the per-dataset lock (the memory helpers directly, the
            # others via datasets.delete_data/empty_dataset). Entering it first
            # would hold a dataset-queue slot while waiting on the lock —
            # the SDK-483 order inversion (canonical: dataset lock -> queue slot).
            if memory_only and data_id is not None:
                result = await _forget_data_memory(data_id, dataset_ref, user)
            elif memory_only:
                result = await _forget_dataset_memory(dataset_ref, user)
            elif data_id is not None:
                result = await _forget_data_item(data_id, dataset_ref, user)
            else:
                result = await _forget_dataset(dataset_ref, user)
            result["history_cleaned"] = clean_history
            return result


async def _forget_everything(user: Any) -> dict:
    """Delete every dataset the user may delete, and the sessions that go with it.

    Cleanup scope:
    - Relational DB (datasets, data records): yes
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Sessions: the same rule as datasets. Every session the user owns, plus
      every session attributed to a dataset the user may delete, whoever ran
      it, goes entirely (cache content, Q&A vectors, dashboard and usage
      rows, via ``delete_sessions``). The user's own sessions are read from
      the cache as well as from ``session_records``, whose rows are written
      best-effort; another user's session is found only through its row,
      since its dataset attribution lives there. Other users' sessions on
      datasets the user cannot delete, the cache's shared key/value rows
      (locks) and the usage logs are never touched: the cache is shared by
      every user and is never pruned here.
    - Search history (queries/results rows): yes — every user's rows on each
      deleted dataset, shared ones included (inside datasets.empty_dataset),
      plus every row this user owns, including rows saved without a dataset
      (before cognee 1.5.0, or a search over several datasets with backend
      access control off). Other users' rows without a dataset stay
    """
    from cognee.api.v1.datasets.datasets import datasets
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.search.operations import delete_user_search_history
    from cognee.modules.session_lifecycle import (
        list_sessions_for_datasets,
        list_sessions_for_user,
    )
    from cognee.modules.session_lifecycle.delete_session import delete_sessions

    # The datasets delete_all removes: the ones the user may delete, not
    # every one they can read.
    deletable_datasets = await get_authorized_existing_datasets([], "delete", user)
    count = len(deletable_datasets)

    # Resolve the sessions before anything is deleted; nothing to list
    # without session memory (CACHING=false).
    session_manager = get_session_manager()
    session_pairs: list[tuple[UUID, str]] = []
    if session_manager.is_available:
        session_pairs.extend(await list_sessions_for_user(user.id))
        session_pairs.extend(
            (user.id, session_id)
            for session_id in await session_manager.list_session_ids(user_id=str(user.id))
        )
        session_pairs.extend(
            await list_sessions_for_datasets([dataset.id for dataset in deletable_datasets])
        )
        session_pairs = list(dict.fromkeys(session_pairs))

    # Before the data, and not caught, as in empty_dataset: if it fails,
    # nothing is deleted yet and the forget can be retried.
    removed = await delete_user_search_history(user.id)
    if removed["queries_deleted"] or removed["results_deleted"]:
        logger.info(
            "forget: removed %d search history queries and %d results for user=%s",
            removed["queries_deleted"],
            removed["results_deleted"],
            user.id,
        )

    # Also before the data, and not caught: the dataset half of the session
    # rule finds its sessions only while the datasets exist, so a failure here
    # must leave them in place for the retry. The deletable datasets' stores
    # are not detagged: delete_all drops them, Q&A vectors included.
    sessions_removed = (
        await delete_sessions(
            session_pairs,
            datasets_being_deleted=[dataset.id for dataset in deletable_datasets],
        )
        if session_pairs
        else 0
    )

    await datasets.delete_all(user=user)

    logger.info(
        "forget: deleted all data for user=%s (%d datasets, %d sessions)",
        user.id,
        count,
        sessions_removed,
    )
    return {
        "datasets_removed": count,
        "sessions_removed": sessions_removed,
        "search_history_removed": True,
        "history_cleaned": True,
        "status": "success",
    }


async def _forget_dataset(dataset_ref: str | UUID, user: Any) -> dict:
    """Delete an entire dataset by name or UUID.

    Cleanup scope:
    - Relational DB (datasets, data records): yes
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Session cache: yes — sessions attributed to the dataset (via
      session_records.dataset_id or the per-dataset default session id)
      are deleted inside datasets.empty_dataset (non-fatal, best-effort)
    - Search history: yes, every user's rows recorded for this dataset, inside
      datasets.empty_dataset — the dataset is gone for everyone it was shared
      with. Rows saved without a dataset stay (see _forget_everything)
    """
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    await datasets.empty_dataset(dataset_id, user=user)

    logger.info("forget: deleted dataset=%s for user=%s", dataset_id, user.id)
    return {
        "dataset_id": str(dataset_id),
        "search_history_removed": True,
        "status": "success",
    }


async def _forget_data_item(data_id: UUID, dataset_ref: str | UUID, user: Any) -> dict:
    """Delete a single data item from a dataset.

    Dataset search history is NOT purged here: rows are attributed to a
    dataset, not to a document, so the only purge available would drop every
    user's history for the whole dataset because one file went. Answers that
    quoted this document's chunks therefore stay until ``forget(dataset=...)``
    or ``forget(everything=True)``; the result says so with
    ``search_history_removed: False``. The caller's own rows go when the
    forget is called with ``clean_history=True`` (handled in ``forget``).
    """
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user, data_id=data_id)

    receipt = await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
        delete_dataset_if_empty=False,
    )

    if receipt["status"] == "not_found":
        # Nothing existed for the id (wrong id, or already deleted): say so, so
        # the log never reads as a deletion that did not happen.
        logger.info(
            "forget: nothing to delete for data_id=%s in dataset=%s for user=%s (not found)",
            data_id,
            dataset_id,
            user.id,
        )
    else:
        logger.info(
            "forget: deleted data_id=%s from dataset=%s for user=%s",
            data_id,
            dataset_id,
            user.id,
        )
    # The receipt keeps its own status, dataset_id and data_id — the latter the
    # *resolved* id, which differs from the input when a legacy id was passed, so
    # it must not be overwritten with the caller's value. Dataset search history
    # is never purged for one item (see the docstring).
    return {**receipt, "search_history_removed": False}


async def _forget_dataset_memory(dataset_ref: str | UUID, user: Any) -> dict:
    """Delete only memory (graph + vector) for a dataset, preserving raw files.

    This allows re-cognifying the dataset with different settings
    (e.g. a new custom prompt or graph model).

    Cleanup scope:
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Session cache: yes — attributed sessions are deleted; QAs in
      dataset-unattributed sessions that used the removed graph ids are
      removed (non-fatal, best-effort)
    - Pipeline status: reset (so cognify re-processes all data)
    - Relational DB (dataset, data records): preserved
    - Raw files: preserved
    """
    from sqlalchemy import select
    from sqlalchemy.orm import attributes as orm_attributes

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.graph.methods.delete_dataset_nodes_and_edges import (
        delete_dataset_nodes_and_edges,
    )

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
    # on this dataset and exclude concurrent deletes. The database context comes
    # AFTER the lock — canonical order (dataset lock -> queue slot), SDK-483.
    async with (
        dataset_lock(dataset_id),
        set_database_global_context_variables(dataset_id, user.id),
    ):
        # 0. The routing rows mirrored the index this dataset is about to lose.
        # They live in the default store, which can be unreachable, so they go
        # before the irreversible step: a failure here leaves the memory in
        # place and the forget can be retried, instead of deleted memory whose
        # summary text survives in the routing collection.
        from cognee.modules.search.dataset_routing import remove_dataset_routes

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            data_ids = [
                record.id
                for record in (
                    await session.execute(select(Data).where(Data.dataset_id == dataset_id))
                )
                .scalars()
                .all()
            ]
        await remove_dataset_routes(dataset_id, data_ids)

        # 1. Delete graph nodes/edges and vector embeddings
        deleted_elements = await delete_dataset_nodes_and_edges(dataset_id, user.id)

        # 1b. Drop sessions attributed to this dataset, then remove tagged
        # QAs from dataset-unattributed sessions (non-fatal).
        try:
            from cognee.modules.session_lifecycle.invalidate_sessions import (
                invalidate_sessions_for_dataset,
                invalidate_sessions_for_deleted_data,
            )

            await invalidate_sessions_for_dataset(dataset_id)
            await invalidate_sessions_for_deleted_data(
                dataset_id,
                deleted_elements.node_ids,
                deleted_elements.edge_ids,
                user_id=user.id,
            )
        except Exception as error:
            logger.warning(
                "forget: session invalidation failed for dataset %s (non-fatal): %s",
                dataset_id,
                error,
                exc_info=True,
            )

        # 1c. The edges are gone, so the evidence rows describing them are stale;
        # the next cognify recaptures them under its own run (non-fatal).
        from cognee.modules.provenance.edge_evidence import delete_edge_evidence

        await delete_edge_evidence(dataset_id)

        # 2. Reset pipeline_status on all data records in this dataset
        async with db_engine.get_async_session() as session:
            data_records = (
                (await session.execute(select(Data).where(Data.dataset_id == dataset_id)))
                .scalars()
                .all()
            )

            dataset_id_str = str(dataset_id)
            for data_record in data_records:
                if not data_record.pipeline_status:
                    continue
                updated = False
                for pipeline_name in list(data_record.pipeline_status.keys()):
                    if dataset_id_str in data_record.pipeline_status[pipeline_name]:
                        del data_record.pipeline_status[pipeline_name][dataset_id_str]
                        updated = True
                if updated:
                    orm_attributes.flag_modified(data_record, "pipeline_status")

            await session.commit()

    logger.info(
        "forget: cleared memory for dataset=%s, user=%s (%d data records reset)",
        dataset_id,
        user.id,
        len(data_records),
    )
    return {
        "dataset_id": str(dataset_id),
        "data_records_reset": len(data_records),
        "status": "success",
    }


async def _forget_data_memory(data_id: UUID, dataset_ref: str | UUID, user: Any) -> dict:
    """Delete only memory (graph + vector) for a single data item, preserving the raw file.

    This allows re-cognifying a specific file with different settings
    without affecting the rest of the dataset.

    Cleanup scope:
    - Graph DB (nodes, edges for this data item): yes
    - Vector DB (embeddings for this data item): yes
    - Session cache: targeted — session entries whose answers used the
      deleted graph elements are removed (non-fatal, best-effort)
    - Pipeline status (for this data item): reset for cognify only
    - Relational DB (data record): preserved
    - Raw file: preserved
    - Search history: untouched — the document stays, and history is
      attributed per dataset, not per document (see _forget_data_item)
    """
    from sqlalchemy import select
    from sqlalchemy.orm import attributes as orm_attributes

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
    # on this dataset and exclude concurrent deletes. The database context comes
    # AFTER the lock — canonical order (dataset lock -> queue slot), SDK-483.
    async with (
        dataset_lock(dataset_id),
        set_database_global_context_variables(dataset_id, user.id),
    ):
        # 0. The routing rows that mirrored the item's bucket and the dataset's
        # root (its text names the item) go first: they live in the default
        # store, which can be unreachable, and a failure there must leave the
        # memory in place for a retry.
        from cognee.modules.search.dataset_routing import remove_dataset_routes

        await remove_dataset_routes(dataset_id, [data_id])

        # 1. Delete graph nodes/edges and vector embeddings for this data item
        deleted_elements = await delete_data_nodes_and_edges(dataset_id, data_id, user.id)

        # 1b. Remove session entries whose answers used the deleted graph
        # elements, so completions stop asserting the removed content (non-fatal).
        try:
            from cognee.modules.session_lifecycle.invalidate_sessions import (
                invalidate_sessions_for_deleted_data,
            )

            await invalidate_sessions_for_deleted_data(
                dataset_id,
                deleted_elements.node_ids,
                deleted_elements.edge_ids,
                user_id=user.id,
            )
        except Exception as error:
            logger.warning(
                "forget: session invalidation failed for data %s in dataset %s (non-fatal): %s",
                data_id,
                dataset_id,
                error,
                exc_info=True,
            )

        # 1c. Drop this item's edge evidence with its edges (non-fatal).
        from cognee.modules.provenance.edge_evidence import delete_edge_evidence

        await delete_edge_evidence(dataset_id, data_id)

        # 1d. The index: the item's bucket and the dataset's root, which names
        # the item, go by their derived ids (the memory delete above does not
        # reach them). Same stores as that delete, so it belongs with it. The
        # root is then written again from what remains, so broad queries keep
        # routing to the dataset; a failure there leaves it to the next improve.
        # A dataset that had no index gets none.
        from cognee.tasks.memify.global_context_index import (
            rebuild_document_index_root,
            remove_document_index,
        )

        if await remove_document_index(dataset_id, [data_id]):
            await rebuild_document_index_root(dataset_id, user)

        # 2. Reset pipeline_status for this data record
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            data_record = (
                (await session.execute(select(Data).where(Data.id == data_id))).scalars().first()
            )

            if data_record and data_record.pipeline_status:
                dataset_id_str = str(dataset_id)
                updated = False
                # Memory-only forget removes the chunks, graph and vectors the
                # indexing and cognify pipelines wrote, so both stamps are reset
                # (a kept indexing stamp would skip the document's early
                # indexing). Keep add status intact.
                for pipeline_name in (INDEXING_PIPELINE_NAME, COGNIFY_PIPELINE_NAME):
                    if dataset_id_str in data_record.pipeline_status.get(pipeline_name, {}):
                        del data_record.pipeline_status[pipeline_name][dataset_id_str]
                        updated = True
                if updated:
                    orm_attributes.flag_modified(data_record, "pipeline_status")
                await session.commit()

    logger.info(
        "forget: cleared memory for data_id=%s in dataset=%s, user=%s",
        data_id,
        dataset_id,
        user.id,
    )
    return {
        "data_id": str(data_id),
        "dataset_id": str(dataset_id),
        "status": "success",
    }


async def _resolve_dataset_id(
    dataset_ref: str | UUID, user: Any, data_id: UUID | None = None
) -> UUID:
    """Resolve a dataset name or UUID to a UUID, with permission check.

    Requires ``delete`` on the dataset. With ``data_id`` (deleting one item),
    ``write`` also passes when the caller added that item (``Data.owner_id``);
    otherwise the ``delete`` denial is raised unchanged, so an author check
    that fails reveals nothing more than a plain denial.
    """
    from cognee.modules.data.exceptions import DatasetNotFoundError
    from cognee.modules.users.exceptions import PermissionDeniedError

    try:
        return await _authorized_dataset_id(dataset_ref, user, "delete")
    except (DatasetNotFoundError, PermissionDeniedError, ValueError):
        if data_id is not None:
            dataset_id = await _authored_item_dataset_id(dataset_ref, user, data_id)
            if dataset_id is not None:
                return dataset_id
        raise


async def _authored_item_dataset_id(
    dataset_ref: str | UUID, user: Any, data_id: UUID
) -> UUID | None:
    """The dataset id if ``user`` has ``write`` on it and added ``data_id`` to it."""
    from cognee.modules.data.exceptions import DatasetNotFoundError
    from cognee.modules.data.methods import get_data
    from cognee.modules.users.exceptions import PermissionDeniedError

    try:
        dataset_id = await _authorized_dataset_id(dataset_ref, user, "write")
    except (DatasetNotFoundError, PermissionDeniedError, ValueError):
        return None
    data = await get_data(user.id, data_id, dataset_id, verify_owner=False)
    if data is None or data.owner_id != user.id:
        return None
    return dataset_id


async def _authorized_dataset_id(dataset_ref: str | UUID, user: Any, permission_type: str) -> UUID:
    """Resolve a dataset name or UUID to a UUID the user holds ``permission_type`` on."""
    if isinstance(dataset_ref, UUID):
        from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset

        dataset = await get_authorized_dataset(user, dataset_ref, permission_type)
        if not dataset:
            raise ValueError(f"Dataset {dataset_ref} not found or not accessible.")
        return dataset.id

    from cognee.modules.data.exceptions import DatasetNotFoundError
    from cognee.modules.data.methods import get_authorized_dataset_by_name

    dataset = await get_authorized_dataset_by_name(dataset_ref, user, permission_type)
    if dataset is None:
        # Same message for missing and unauthorized: the name lookup returns None
        # for both, and distinguishing them would leak which dataset names exist.
        raise DatasetNotFoundError(message=f"Dataset '{dataset_ref}' not found or not accessible.")
    return dataset.id
