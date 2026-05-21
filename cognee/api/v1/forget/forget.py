from uuid import UUID
from typing import Optional, Union

from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability import (
    new_span,
    COGNEE_DATASET_NAME,
    COGNEE_FORGET_TARGET,
    COGNEE_RESULT_COUNT,
)

logger = get_logger("forget")


async def forget(
    *,
    data_id: Optional[UUID] = None,
    dataset: Optional[Union[str, UUID]] = None,
    everything: bool = False,
    memory_only: bool = False,
    user=None,
) -> dict:
    """Remove data from the knowledge graph.

    Unified deletion command that replaces the separate prune/delete/
    empty_dataset APIs with a single mental model.

    Usage patterns::

        # Forget a specific data item from a dataset
        await cognee.forget(data_id=data_id, dataset=dataset_id)

        # Forget an entire dataset (all data + graph nodes + vector entries)
        await cognee.forget(dataset="scientists")

        # Forget everything the current user owns
        await cognee.forget(everything=True)

        # Forget only memory (graph + vector) for a dataset (keep raw files)
        await cognee.forget(dataset="scientists", memory_only=True)

        # Forget only memory for a single file in a dataset
        await cognee.forget(dataset="scientists", data_id=data_id, memory_only=True)

    Args:
        data_id: UUID of a specific data item to remove.
            Requires ``dataset`` to also be set.
        dataset: Dataset name or UUID. When set alone, deletes the
            entire dataset. When set with ``data_id``, deletes that
            item from this dataset.
        everything: If True, delete all datasets and data the user owns.
            Ignores ``data_id`` and ``dataset``.
        memory_only: If True (requires ``dataset``), delete only memory
            (graph nodes/edges and vector embeddings) and reset pipeline status.
            Raw files and data records are preserved so the dataset can
            be re-cognified with different settings.
        user: User context. Resolved to default user when None.

    Returns:
        Dict with deletion summary: items removed, datasets removed.
    """
    from cognee.shared.utils import send_telemetry
    from cognee import __version__ as cognee_version

    if everything:
        target = "everything"
    elif memory_only and data_id:
        target = "data_item_memory_only"
    elif memory_only and dataset:
        target = "dataset_memory_only"
    elif data_id:
        target = "data_item"
    elif dataset:
        target = "dataset"
    else:
        target = "unknown"

    send_telemetry(
        "cognee.forget",
        user if user and hasattr(user, "id") else "sdk",
        additional_properties={
            "target": target,
            "dataset": str(dataset) if dataset else "",
            "data_id": str(data_id) if data_id else "",
            "cognee_version": cognee_version,
        },
    )

    with new_span("cognee.api.forget") as span:
        span.set_attribute(COGNEE_FORGET_TARGET, target)
        if dataset:
            span.set_attribute(COGNEE_DATASET_NAME, str(dataset))

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            result = await client.forget(
                data_id=data_id, dataset=dataset, everything=everything, memory_only=memory_only
            )
            span.set_attribute(
                COGNEE_RESULT_COUNT,
                result.get("datasets_removed", 0) if isinstance(result, dict) else 0,
            )
            return result

        from cognee.modules.users.methods import get_default_user

        # In case there is no database, forget will fail when getting a user
        from cognee.low_level import setup

        await setup()

        if user is None:
            user = await get_default_user()

        async with set_database_global_context_variables(dataset, user.id):
            if everything:
                result = await _forget_everything(user)
                span.set_attribute(COGNEE_RESULT_COUNT, result.get("datasets_removed", 0))
                return result

            if memory_only:
                if dataset is None:
                    raise ValueError("memory_only requires dataset to be specified.")
                if data_id is not None:
                    return await _forget_data_memory(data_id, dataset, user)
                return await _forget_dataset_memory(dataset, user)

            if dataset is not None and data_id is not None:
                return await _forget_data_item(data_id, dataset, user)

            if dataset is not None:
                return await _forget_dataset(dataset, user)

            if data_id is not None:
                raise ValueError("data_id requires dataset to be specified.")

            raise ValueError("Specify dataset, data_id+dataset, or everything=True.")


async def _forget_everything(user) -> dict:
    """Delete all datasets, data, and session cache owned by the user.

    Cleanup scope:
    - Relational DB (datasets, data records): yes
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Session cache (Redis/FS): yes (full prune)
    """
    from cognee.api.v1.datasets.datasets import datasets

    user_datasets = await datasets.list_datasets(user=user)
    count = len(user_datasets)

    await datasets.delete_all(user=user)

    # Clean up session cache (Redis or filesystem)
    try:
        from cognee.infrastructure.databases.cache import get_cache_config
        from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

        cache_config = get_cache_config()
        if cache_config.caching or cache_config.usage_logging:
            cache_engine = get_cache_engine()
            if cache_engine is not None:
                await cache_engine.prune()
    except Exception as e:
        logger.warning("forget: session cache cleanup failed (non-fatal): %s", e)

    logger.info("forget: deleted all data for user=%s (%d datasets)", user.id, count)
    return {"datasets_removed": count, "status": "success"}


async def _forget_dataset(dataset_ref: Union[str, UUID], user) -> dict:
    """Delete an entire dataset by name or UUID.

    Cleanup scope:
    - Relational DB (datasets, data records): yes
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Session cache: no (sessions are keyed by user_id+session_id,
      not by dataset — targeted cleanup requires tagging sessions
      with dataset_id, which is a future enhancement)
    """
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    await datasets.empty_dataset(dataset_id, user=user)

    logger.info("forget: deleted dataset=%s for user=%s", dataset_id, user.id)
    return {"dataset_id": str(dataset_id), "status": "success"}


async def _forget_data_item(data_id: UUID, dataset_ref: Union[str, UUID], user) -> dict:
    """Delete a single data item from a dataset."""
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
        delete_dataset_if_empty=False,
    )

    logger.info(
        "forget: deleted data_id=%s from dataset=%s for user=%s",
        data_id,
        dataset_id,
        user.id,
    )
    return {"data_id": str(data_id), "dataset_id": str(dataset_id), "status": "success"}


async def _forget_dataset_memory(dataset_ref: Union[str, UUID], user) -> dict:
    """Delete only memory (graph + vector) for a dataset, preserving raw files.

    This allows re-cognifying the dataset with different settings
    (e.g. a new custom prompt or graph model).

    Cleanup scope:
    - Graph DB (nodes, edges): yes
    - Vector DB (embeddings): yes
    - Pipeline status: reset (so cognify re-processes all data)
    - Relational DB (dataset, data records): preserved
    - Raw files: preserved
    """
    from sqlalchemy import select
    from sqlalchemy.orm import attributes as orm_attributes

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.data.models.DatasetData import DatasetData
    from cognee.modules.graph.methods.delete_dataset_nodes_and_edges import (
        delete_dataset_nodes_and_edges,
    )
    from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
        reset_dataset_pipeline_run_status,
    )

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    # 1. Delete graph nodes/edges and vector embeddings
    await delete_dataset_nodes_and_edges(dataset_id, user.id)

    # 2. Reset pipeline_status on all data records in this dataset
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_ids_query = select(DatasetData.data_id).where(DatasetData.dataset_id == dataset_id)
        data_records = (
            (await session.execute(select(Data).where(Data.id.in_(data_ids_query)))).scalars().all()
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

    # 3. Reset dataset-level pipeline run status so cached cognify runs can execute again.
    await reset_dataset_pipeline_run_status(
        dataset_id=dataset_id,
        user=user,
        pipeline_names=["cognify_pipeline"],
    )

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


async def _forget_data_memory(data_id: UUID, dataset_ref: Union[str, UUID], user) -> dict:
    """Delete only memory (graph + vector) for a single data item, preserving the raw file.

    This allows re-cognifying a specific file with different settings
    without affecting the rest of the dataset.

    Cleanup scope:
    - Graph DB (nodes, edges for this data item): yes
    - Vector DB (embeddings for this data item): yes
    - Pipeline status (for this data item): reset for cognify only
    - Relational DB (data record): preserved
    - Raw file: preserved
    """
    from sqlalchemy import select
    from sqlalchemy.orm import attributes as orm_attributes

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    # 1. Delete graph nodes/edges and vector embeddings for this data item
    await delete_data_nodes_and_edges(dataset_id, data_id, user.id)

    # 2. Reset pipeline_status for this data record
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_record = (
            (await session.execute(select(Data).where(Data.id == data_id))).scalars().first()
        )

        if data_record and data_record.pipeline_status:
            dataset_id_str = str(dataset_id)
            updated = False
            # Memory-only forget removes cognify artifacts (graph/vector), so only
            # cognify_pipeline status should be reset. Keep add status intact.
            if (
                "cognify_pipeline" in data_record.pipeline_status
                and dataset_id_str in data_record.pipeline_status["cognify_pipeline"]
            ):
                del data_record.pipeline_status["cognify_pipeline"][dataset_id_str]
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


async def _resolve_dataset_id(dataset_ref: Union[str, UUID], user) -> UUID:
    """Resolve a dataset name or UUID to a UUID, with permission check."""
    if isinstance(dataset_ref, UUID):
        from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset

        dataset = await get_authorized_dataset(user, dataset_ref, "delete")
        if not dataset:
            raise ValueError(f"Dataset {dataset_ref} not found or not accessible.")
        return dataset.id

    from cognee.modules.data.methods import get_authorized_dataset_by_name

    dataset = await get_authorized_dataset_by_name(dataset_ref, user, "delete")
    return dataset.id
