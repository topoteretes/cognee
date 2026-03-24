from datetime import datetime, timezone
from typing import Optional
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


async def status(
    datasets: Optional[list[str]] = None,
    *,
    user=None,
) -> list[DatasetStatus]:
    """Return processing status for the user's datasets.

    Args:
        datasets: Optional list of dataset names to filter by.
        user: User context (resolved to default user when ``None``).

    Returns:
        List of ``DatasetStatus`` objects, one per dataset.
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

    # Query 1: Bulk-load all Data items joined through DatasetData
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(
                    DatasetData.dataset_id,
                    Data.pipeline_status,
                    Data.updated_at,
                )
                .join(Data, DatasetData.data_id == Data.id)
                .where(DatasetData.dataset_id.in_(dataset_ids))
            )
        ).all()

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
                last_data_updated_at=(
                    datetime.fromtimestamp(bucket["last_updated"] / 1000, tz=timezone.utc)
                    if bucket["last_updated"] is not None
                    else None
                ),
                version=version_map.get(did, 0),
            )
        )

    return results
