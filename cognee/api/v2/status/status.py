from datetime import datetime
from typing import Optional, Union
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Data import Data
from cognee.modules.data.models.DatasetData import DatasetData
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status


class DatasetStatus(BaseModel):
    """Status snapshot for a single dataset."""

    dataset_id: UUID
    dataset_name: str
    owner_id: UUID
    item_count: int
    completed: int
    pending: int
    add_pipeline_status: Optional[str] = None
    cognify_pipeline_status: Optional[str] = None
    dataset_created_at: datetime
    dataset_updated_at: Optional[datetime] = None
    last_data_updated_at: Optional[datetime] = None
    version: int


class DataItemInfo(BaseModel):
    """Per-item status within a dataset."""

    data_id: UUID
    dataset_id: UUID
    dataset_name: str
    name: str
    status: str  # "completed", "errored", "pending"
    error: Optional[str] = None
    content_hash: str
    ingested_at: datetime
    updated_at: Optional[datetime] = None


async def status(
    datasets: Optional[list[str]] = None,
    *,
    items: bool = False,
    since: Optional[datetime] = None,
    user=None,
) -> Union[list[DatasetStatus], list[DataItemInfo]]:
    """Return processing status for the user's datasets.

    When ``items=False`` (default), returns one ``DatasetStatus`` per
    dataset with aggregate counts.  When ``items=True``, returns one
    ``DataItemInfo`` per data item showing per-file status, including
    error messages for failed items.

    Args:
        datasets: Optional list of dataset names to filter by.
        items: If True, return per-item detail instead of aggregates.
        since: Only include items created at or after this timestamp.
            Applies to both aggregate counts and item-level detail.
        user: User context (resolved to default user when ``None``).

    Returns:
        List of ``DatasetStatus`` or ``DataItemInfo`` objects.
    """
    from cognee.modules.users.methods.get_default_user import get_default_user
    from cognee.modules.data.methods.get_authorized_existing_datasets import (
        get_authorized_existing_datasets,
    )

    if user is None:
        user = await get_default_user()

    existing_datasets = await get_authorized_existing_datasets(
        datasets or [], permission_type="read", user=user
    )

    if not existing_datasets:
        return []

    dataset_map = {ds.id: ds for ds in existing_datasets}
    dataset_ids = list(dataset_map.keys())

    if items:
        return await _item_level_status(dataset_ids, dataset_map, since=since)

    # Query 1: Bulk-load all Data items joined through DatasetData
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        stmt = (
            select(
                DatasetData.dataset_id,
                Data.pipeline_status,
                Data.updated_at,
            )
            .join(Data, DatasetData.data_id == Data.id)
            .where(DatasetData.dataset_id.in_(dataset_ids))
        )
        if since is not None:
            stmt = stmt.where(Data.created_at >= since)
        rows = (await session.execute(stmt)).all()

    # Aggregate per dataset
    agg: dict[UUID, dict] = {
        did: {"item_count": 0, "completed": 0, "last_updated": None} for did in dataset_ids
    }

    for dataset_id, pipeline_status, updated_at in rows:
        bucket = agg[dataset_id]
        bucket["item_count"] += 1

        # Check if cognify_pipeline completed for this dataset
        if (
            pipeline_status
            and isinstance(pipeline_status, dict)
            and pipeline_status.get("cognify_pipeline", {}).get(str(dataset_id))
            == "DATA_ITEM_PROCESSING_COMPLETED"
        ):
            bucket["completed"] += 1

        if updated_at is not None:
            if bucket["last_updated"] is None or updated_at > bucket["last_updated"]:
                bucket["last_updated"] = updated_at

    # Query 2: Pipeline run statuses (reuses existing helper)
    add_statuses = await get_pipeline_status(dataset_ids, "add_pipeline")
    cognify_statuses = await get_pipeline_status(dataset_ids, "cognify_pipeline")

    # Query 3: Version counts (number of PipelineRun records per dataset)
    from cognee.modules.pipelines.models.PipelineRun import PipelineRun
    from sqlalchemy import func

    async with db_engine.get_async_session() as session:
        version_rows = (
            await session.execute(
                select(
                    PipelineRun.dataset_id,
                    func.count().label("version"),
                )
                .where(PipelineRun.dataset_id.in_(dataset_ids))
                .group_by(PipelineRun.dataset_id)
            )
        ).all()

    version_map = {row.dataset_id: row.version for row in version_rows}

    # Build result
    results = []
    for did in dataset_ids:
        ds = dataset_map[did]
        bucket = agg[did]
        item_count = bucket["item_count"]
        completed = bucket["completed"]

        add_status = add_statuses.get(str(did))
        cognify_status = cognify_statuses.get(str(did))

        results.append(
            DatasetStatus(
                dataset_id=did,
                dataset_name=ds.name,
                owner_id=ds.owner_id,
                item_count=item_count,
                completed=completed,
                pending=item_count - completed,
                add_pipeline_status=add_status.value if add_status else None,
                cognify_pipeline_status=cognify_status.value if cognify_status else None,
                dataset_created_at=ds.created_at,
                dataset_updated_at=ds.updated_at,
                last_data_updated_at=bucket["last_updated"],
                version=version_map.get(did, 0),
            )
        )

    return results


async def _item_level_status(dataset_ids, dataset_map, since=None):
    """Return per-item DataItemInfo for the given datasets."""
    from cognee.modules.pipelines.models.PipelineRun import PipelineRun, PipelineRunStatus

    db_engine = get_relational_engine()

    # Load all data items across requested datasets
    async with db_engine.get_async_session() as session:
        stmt = (
            select(
                DatasetData.dataset_id,
                Data.id,
                Data.name,
                Data.content_hash,
                Data.pipeline_status,
                Data.created_at,
                Data.updated_at,
            )
            .join(Data, DatasetData.data_id == Data.id)
            .where(DatasetData.dataset_id.in_(dataset_ids))
        )
        if since is not None:
            stmt = stmt.where(Data.created_at >= since)
        rows = (await session.execute(stmt)).all()

    if not rows:
        return []

    # Collect dataset_ids that have errored items so we can look up error messages
    errored_dataset_ids = set()
    for dataset_id, _, _, _, pipeline_status, _, _ in rows:
        if not pipeline_status or not isinstance(pipeline_status, dict):
            continue
        cognify = pipeline_status.get("cognify_pipeline", {})
        if cognify.get(str(dataset_id)) != "DATA_ITEM_PROCESSING_COMPLETED":
            errored_dataset_ids.add(dataset_id)

    # Load latest errored PipelineRun per dataset for error messages
    error_map = {}
    if errored_dataset_ids:
        async with db_engine.get_async_session() as session:
            error_runs = (
                (
                    await session.execute(
                        select(PipelineRun)
                        .where(PipelineRun.dataset_id.in_(list(errored_dataset_ids)))
                        .where(PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_ERRORED)
                        .order_by(PipelineRun.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )

        # Keep only the most recent error per dataset
        for run in error_runs:
            if run.dataset_id not in error_map:
                error_map[run.dataset_id] = run.run_info

    # Build per-item results
    results = []
    for dataset_id, data_id, name, content_hash, pipeline_status, created_at, updated_at in rows:
        ds = dataset_map[dataset_id]
        ds_id_str = str(dataset_id)

        # Determine item status from pipeline_status JSON
        cognify_status = None
        if pipeline_status and isinstance(pipeline_status, dict):
            cognify_status = pipeline_status.get("cognify_pipeline", {}).get(ds_id_str)

        if cognify_status == "DATA_ITEM_PROCESSING_COMPLETED":
            item_status = "completed"
            error = None
        elif dataset_id in error_map:
            item_status = "errored"
            run_info = error_map[dataset_id]
            error = str(run_info) if run_info else None
        else:
            item_status = "pending"
            error = None

        results.append(
            DataItemInfo(
                data_id=data_id,
                dataset_id=dataset_id,
                dataset_name=ds.name,
                name=name,
                status=item_status,
                error=error,
                content_hash=content_hash,
                ingested_at=created_at,
                updated_at=updated_at,
            )
        )

    return results
