import asyncio
from uuid import UUID

from cognee.api.v1.datasets.dto import DataDTO
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods import (
    get_authorized_dataset,
    get_authorized_existing_datasets,
    get_dataset_data,
    has_dataset_data,
)
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
from cognee.modules.pipelines.methods import get_latest_pipeline_runs_by_datasets
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import (
    get_pipeline_progress,
    get_pipeline_status,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


# Deprecated status alias: code repositories are built under cognify_pipeline
# now (before, they ran under this name), so only pre-upgrade runs and custom
# pipelines still write runs under it. A status query for it reports the
# dataset's latest code_graph_pipeline or cognify_pipeline run, preferring one
# still in progress and otherwise the newer, so legacy pollers (e.g. cognee-mcp)
# see a post-upgrade build finish even on a dataset whose last pre-upgrade build
# left a code_graph_pipeline run.
LEGACY_CODE_GRAPH_PIPELINE = "code_graph_pipeline"
COGNIFY_PIPELINE = "cognify_pipeline"

_UNFINISHED_RUN_STATUSES = frozenset(
    {
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        PipelineRunStatus.DATASET_PROCESSING_INITIATED,
    }
)


def _alias_run_rank(run) -> tuple[bool, object]:
    """Sort key for the alias: an unfinished run beats a terminal one, then the newer wins."""
    return (run.status in _UNFINISHED_RUN_STATUSES, run.created_at)


async def _resolve_legacy_alias(dataset_ids: list[UUID]) -> dict[UUID, str]:
    """Per dataset, the pipeline whose latest run answers a ``code_graph_pipeline``
    status query: a run still in progress (STARTED/INITIATED) beats a terminal
    one, then the newer by ``created_at`` wins; a full tie goes to
    cognify_pipeline. Datasets with no run under either name are left out.
    """
    legacy_runs = await get_latest_pipeline_runs_by_datasets(
        dataset_ids, LEGACY_CODE_GRAPH_PIPELINE
    )
    cognify_runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, COGNIFY_PIPELINE)

    resolved = {}
    for dataset_id in dataset_ids:
        legacy_run = legacy_runs.get(dataset_id)
        cognify_run = cognify_runs.get(dataset_id)
        if legacy_run is None and cognify_run is None:
            continue
        if cognify_run is None or (
            legacy_run is not None and _alias_run_rank(legacy_run) > _alias_run_rank(cognify_run)
        ):
            resolved[dataset_id] = LEGACY_CODE_GRAPH_PIPELINE
        else:
            resolved[dataset_id] = COGNIFY_PIPELINE
    return resolved


_legacy_alias_warned = False


def _warn_legacy_code_graph_alias() -> None:
    """Warn once per process: pollers hit the alias every few seconds."""
    global _legacy_alias_warned
    if _legacy_alias_warned:
        return
    _legacy_alias_warned = True
    logger.warning(
        "Pipeline status for '%s' is deprecated: code repositories are built under"
        " cognify_pipeline, so '%s' reports its own or cognify_pipeline's latest run"
        " (an in-progress run first, else the newer). Query cognify_pipeline instead."
        " Logged once per process.",
        LEGACY_CODE_GRAPH_PIPELINE,
        LEGACY_CODE_GRAPH_PIPELINE,
    )


async def _fetch_with_legacy_fallback(fetch, dataset_ids: list[UUID], pipeline_name: str):
    if pipeline_name != LEGACY_CODE_GRAPH_PIPELINE:
        return await fetch(dataset_ids, pipeline_name=pipeline_name)

    _warn_legacy_code_graph_alias()
    resolved = await _resolve_legacy_alias(dataset_ids)
    statuses = {}
    for resolved_name in (LEGACY_CODE_GRAPH_PIPELINE, COGNIFY_PIPELINE):
        group = [dataset_id for dataset_id, name in resolved.items() if name == resolved_name]
        if group:
            statuses.update(await fetch(group, pipeline_name=resolved_name))
    return statuses


async def get_processing_status_with_legacy_alias(dataset_id: UUID, pipeline_name: str) -> dict:
    """get_dataset_processing_status, honouring the ``code_graph_pipeline`` alias.

    Items built before the upgrade carry only the ``code_graph_pipeline`` stamp
    and items built after it only the ``cognify_pipeline`` one, so for the alias
    an item counts as completed when either stamp is. The dataset's rows are
    read once.
    """
    from cognee.modules.data import methods as data_methods

    if pipeline_name != LEGACY_CODE_GRAPH_PIPELINE:
        return await data_methods.get_dataset_processing_status(
            dataset_id, pipeline_name=pipeline_name
        )

    _warn_legacy_code_graph_alias()
    return await data_methods.get_dataset_processing_status(
        dataset_id,
        pipeline_name=COGNIFY_PIPELINE,
        also_completed_by=(LEGACY_CODE_GRAPH_PIPELINE,),
    )


async def _fan_out_by_pipeline(dataset_ids: list[UUID], pipeline_names: list[str] | None, fetch):
    """Shared flat/nested shaping for get_status and get_progress.

    ``fetch`` is get_pipeline_status or get_pipeline_progress — only the
    per-dataset value type differs (a bare status vs. {status, progress});
    the flat-vs-nested decision based on how many pipeline names were
    requested is identical either way, so it lives here once.
    """
    # Backward-compatible default: cognify-only flat map.
    if not pipeline_names:
        return await fetch(dataset_ids, pipeline_name=COGNIFY_PIPELINE)

    # Preserve order while removing duplicates.
    requested_pipelines = list(dict.fromkeys(pipeline_names))

    # For one pipeline, keep flat shape.
    if len(requested_pipelines) == 1:
        return await _fetch_with_legacy_fallback(fetch, dataset_ids, requested_pipelines[0])

    # For multiple pipelines, return nested shape.
    statuses_by_dataset = {str(dataset_id): {} for dataset_id in dataset_ids}
    for pipeline_name in requested_pipelines:
        pipeline_status = await _fetch_with_legacy_fallback(fetch, dataset_ids, pipeline_name)
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
        logger.warning(
            "Session invalidation after dataset delete failed (non-fatal): %s", error, exc_info=True
        )


async def _invalidate_sessions_for_deleted_data_nonfatal(
    dataset_id: UUID,
    deleted_elements: DeletedGraphElements | None,
    user_id: UUID | None = None,
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
        logger.warning(
            "Session invalidation after data delete failed (non-fatal): %s", error, exc_info=True
        )


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
    async def list_datasets(user: User | None = None):
        if user is None:
            user = await get_default_user()

        return await get_authorized_existing_datasets([], "read", user)

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: UUID, user: User | None = None):
        # Route to the remote instance when connected via serve(): the dataset
        # lives on the server, so the local store would report it missing.
        # Rows are parsed through the same DTO the server serializes them
        # with, so callers read the same attributes (``row.id`` as a UUID,
        # ``row.mime_type`` not ``mimeType``) in both modes.
        from cognee.api.v1.serve.state import get_remote_client
        from cognee.modules.data.methods import get_dataset_data

        client = get_remote_client()
        if client is not None:
            return [DataDTO.model_validate(row) for row in await client.list_data(dataset_id)]

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await get_dataset_data(dataset.id, order_by="created_at")

    @staticmethod
    async def has_data(dataset_id: str, user: User | None = None) -> bool:
        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await has_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID], pipeline_names: list[str] | None = None) -> dict:
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_status)

    @staticmethod
    async def get_progress(
        dataset_ids: list[UUID], pipeline_names: list[str] | None = None
    ) -> dict:
        """Same flat-or-nested shape as get_status, but each value is
        {status, progress} instead of a bare status. A separate method
        rather than a flag on get_status, so get_status's response shape
        never depends on how it was called.
        """
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_progress)

    @staticmethod
    async def empty_dataset(dataset_id: UUID, user: User | None = None):
        from cognee.modules.data.methods import delete_data, delete_dataset

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id, "delete")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
        # on this dataset and exclude concurrent deletes.
        async with (
            dataset_lock(dataset.id),
            set_database_global_context_variables(dataset.id, dataset.owner_id),
        ):
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
        user: User | None = None,
        mode: str = "soft",  # mode is there for backwards compatibility. Don't use "hard", it is dangerous.
        delete_dataset_if_empty: bool = False,  # if this flag is True, delete the whole dataset if it is left empty after data deletion
    ):
        async with record_operation(
            "delete", user=user, dataset_id=dataset_id
        ) as operation_context:
            from cognee.modules.data.methods import delete_data, delete_dataset, get_data

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
                        elif hasattr(await get_graph_engine(), "get_document_subgraph"):
                            deleted_elements = await legacy_delete(data, "soft")
                        else:
                            # An unmarked graph on an adapter without the legacy
                            # subgraph API (the Postgres demo) holds no legacy
                            # data: it is a document that was never cognified,
                            # so there is nothing in the graph to delete.
                            deleted_elements = DeletedGraphElements()

                    await _invalidate_sessions_for_deleted_data_nonfatal(
                        dataset.id, deleted_elements, user.id
                    )

                    await delete_data(data, dataset_id)

                    dataset_data = await get_dataset_data(dataset.id)
                    if not dataset_data and delete_dataset_if_empty:
                        await delete_dataset(dataset)

            return {"status": "success"}

    @staticmethod
    async def delete_all(user: User | None = None):
        if not user:
            user = await get_default_user()

        user_datasets = await get_authorized_existing_datasets([], "delete", user)

        for dataset in user_datasets:
            await datasets.empty_dataset(dataset.id, user)
