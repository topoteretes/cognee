import asyncio
from uuid import UUID
from typing import Optional

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.data.methods import get_dataset_data, has_dataset_data
from cognee.modules.data.methods import get_authorized_dataset, get_authorized_existing_datasets
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.graph.methods import (
    delete_data_nodes_and_edges,
    delete_dataset_nodes_and_edges,
    has_data_related_nodes,
    legacy_delete,
    try_delete_data_by_graph_provenance,
)
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.operations.get_pipeline_status import (
    get_pipeline_status,
    get_pipeline_progress,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def _fan_out_by_pipeline(dataset_ids: list[UUID], pipeline_names: Optional[list[str]], fetch):
    """Shared flat/nested shaping for get_status and get_progress.

    ``fetch`` is get_pipeline_status or get_pipeline_progress — only the
    per-dataset value type differs (a bare status vs. {status, progress});
    the flat-vs-nested decision based on how many pipeline names were
    requested is identical either way, so it lives here once.
    """
    # Backward-compatible default: cognify-only flat map.
    if not pipeline_names:
        return await fetch(dataset_ids, pipeline_name="cognify_pipeline")

    # Preserve order while removing duplicates.
    requested_pipelines = list(dict.fromkeys(pipeline_names))

    # For one pipeline, keep flat shape.
    if len(requested_pipelines) == 1:
        return await fetch(dataset_ids, pipeline_name=requested_pipelines[0])

    # For multiple pipelines, return nested shape.
    statuses_by_dataset = {str(dataset_id): {} for dataset_id in dataset_ids}
    for pipeline_name in requested_pipelines:
        pipeline_status = await fetch(dataset_ids, pipeline_name=pipeline_name)
        for dataset_id, status in pipeline_status.items():
            statuses_by_dataset.setdefault(dataset_id, {})[pipeline_name] = status

    return statuses_by_dataset


async def _invalidate_sessions_for_dataset_nonfatal(dataset_id: UUID) -> None:
    """Drop sessions attributed to a deleted dataset. Never fails the delete."""
    try:
        from cognee.modules.session_lifecycle.invalidate_sessions import (
            invalidate_sessions_for_dataset,
        )

        await invalidate_sessions_for_dataset(dataset_id)
    except Exception as error:
        logger.warning("Session invalidation after dataset delete failed (non-fatal): %s", error)


async def _invalidate_sessions_for_deleted_data_nonfatal(
    dataset_id: UUID,
    deleted_elements: Optional[DeletedGraphElements],
    user_id: Optional[UUID] = None,
) -> None:
    """Remove session entries that used the deleted elements. Never fails the delete."""
    if deleted_elements is None:
        return
    try:
        from cognee.modules.session_lifecycle.invalidate_sessions import (
            invalidate_sessions_for_deleted_data,
        )

        await invalidate_sessions_for_deleted_data(
            dataset_id,
            deleted_elements.node_ids,
            deleted_elements.edge_ids,
            user_id=user_id,
        )
    except Exception as error:
        logger.warning("Session invalidation after data delete failed (non-fatal): %s", error)


class datasets:
    """
    Dataset management namespace for Cognee.

    All methods are static and provide operations for listing, inspecting,
    and deleting datasets and the data items within them.

    Example:
        ```python
        import cognee

        # List all accessible datasets
        all_datasets = await cognee.datasets.list_datasets()

        # Check cognify processing status for datasets
        status = await cognee.datasets.get_status([dataset_id])

        # Delete a specific data item from a dataset
        await cognee.datasets.delete_data(dataset_id=dataset_id, data_id=data_id)
        ```
    """

    @staticmethod
    async def list_datasets(user: Optional[User] = None):
        if user is None:
            user = await get_default_user()

        return await get_authorized_existing_datasets([], "read", user)

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import get_dataset_data

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await get_dataset_data(dataset.id)

    @staticmethod
    async def has_data(dataset_id: str, user: Optional[User] = None) -> bool:
        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await has_dataset_data(dataset.id)

    @staticmethod
    async def get_status(
        dataset_ids: list[UUID], pipeline_names: Optional[list[str]] = None
    ) -> dict:
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_status)

    @staticmethod
    async def get_progress(
        dataset_ids: list[UUID], pipeline_names: Optional[list[str]] = None
    ) -> dict:
        """Same flat-or-nested shape as get_status, but each value is
        {status, progress} instead of a bare status. A separate method
        rather than a flag on get_status, so get_status's response shape
        never depends on how it was called.
        """
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_progress)

    @staticmethod
    async def empty_dataset(dataset_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import delete_data, delete_dataset

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id, "delete")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
        # on this dataset and exclude concurrent deletes.
        async with dataset_lock(dataset.id):
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                deleted_elements = await delete_dataset_nodes_and_edges(dataset_id, user.id)

                # Session memory derived from this dataset would keep asserting
                # the deleted content (stale QA replay / session-context leak),
                # so drop the attributed sessions with the dataset.
                await _invalidate_sessions_for_dataset_nonfatal(dataset.id)
                await _invalidate_sessions_for_deleted_data_nonfatal(
                    dataset.id, deleted_elements, user.id
                )

                # delete_dataset removes the dataset's scoped Data rows
                # (files refcounted by raw_data_location) with the record.
                result = await delete_dataset(dataset)

        return result

    @staticmethod
    async def delete_data(
        dataset_id: UUID,
        data_id: UUID,
        user: Optional[User] = None,
        mode: str = "soft",  # mode is there for backwards compatibility. Don't use "hard", it is dangerous.
        delete_dataset_if_empty: bool = False,  # if this flag is True, delete the whole dataset if it is left empty after data deletion
    ):
        async with record_operation(
            "delete", user=user, dataset_id=dataset_id
        ) as operation_context:
            from cognee.modules.data.methods import delete_data, get_data, delete_dataset

            if not user:
                user = await get_default_user()
            operation_context.set_user(user)

            try:
                dataset = await get_authorized_dataset(user, dataset_id, "delete")
            except PermissionDeniedError:
                raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

            if not dataset:
                raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

            # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
            # on this dataset and exclude concurrent deletes.
            async with dataset_lock(dataset.id):
                # Every id ever issued keeps resolving: exact id first, then the
                # recorded original of a backfill-split row (legacy_id).
                from cognee.modules.data.methods import resolve_data_id

                resolved_id = await resolve_data_id(dataset.id, data_id)
                if resolved_id is not None:
                    data_id = resolved_id

                dataset_data = [
                    data for data in await get_dataset_data(dataset.id) if data.id == data_id
                ]

                data = dataset_data[0] if len(dataset_data) > 0 else None

                if not data:
                    # If data is not found in the system, user is using a custom graph model.
                    async with set_database_global_context_variables(dataset_id, dataset.owner_id):
                        deleted_elements = await delete_data_nodes_and_edges(
                            dataset_id, data_id, user.id
                        )
                        await _invalidate_sessions_for_deleted_data_nonfatal(
                            dataset.id, deleted_elements, user.id
                        )

                        dataset_data = await get_dataset_data(dataset.id)
                        if not dataset_data and delete_dataset_if_empty:
                            await delete_dataset(dataset)

                    return {"status": "success"}

                if str(data.dataset_id) != str(dataset_id):
                    raise UnauthorizedDataAccessError(f"Data {data_id} not accessible.")

                async with set_database_global_context_variables(dataset_id, dataset.owner_id):
                    # Delete mode is exclusive: ledger rows imply the relational-ledger
                    # path; only ledger-free data probes the graph marker to distinguish
                    # graph-provenance data from legacy data without graph ownership.
                    if await has_data_related_nodes(dataset_id, data_id):
                        deleted_elements = await delete_data_nodes_and_edges(
                            dataset_id, data_id, user.id
                        )
                    else:
                        provenance_result = await try_delete_data_by_graph_provenance(
                            dataset_id, data_id
                        )
                        if provenance_result is not None:
                            deleted_elements = DeletedGraphElements.from_source_ref_removal(
                                provenance_result
                            )
                        else:
                            deleted_elements = await legacy_delete(data, "soft")

                    await _invalidate_sessions_for_deleted_data_nonfatal(
                        dataset.id, deleted_elements, user.id
                    )

                    await delete_data(data, dataset_id)

                    dataset_data = await get_dataset_data(dataset.id)
                    if not dataset_data and delete_dataset_if_empty:
                        await delete_dataset(dataset)

            return {"status": "success"}

    @staticmethod
    async def delete_all(user: Optional[User] = None):
        if not user:
            user = await get_default_user()

        user_datasets = await get_authorized_existing_datasets([], "delete", user)

        for dataset in user_datasets:
            await datasets.empty_dataset(dataset.id, user)
