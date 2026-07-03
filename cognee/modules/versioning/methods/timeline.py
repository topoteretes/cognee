"""Resolve "as of T" against the pipeline-run ledger.

The version timeline of a dataset is its ordered sequence of *completed*
pipeline runs. One logical run writes several ``PipelineRun`` status rows
(INITIATED/STARTED/COMPLETED) sharing a ``pipeline_run_id``, so the ledger
unit here is the ``pipeline_run_id`` deduplicated over rows with
``DATASET_PROCESSING_COMPLETED`` status.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.versioning.models import DatasetSnapshot


def ensure_utc(value: datetime) -> datetime:
    """Coerce a naive datetime (SQLite round-trips lose tzinfo) to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def resolve_as_of_time(dataset_id: UUID, as_of: Union[str, datetime]) -> datetime:
    """Resolve an ``as_of`` argument (snapshot name or datetime) to a UTC datetime."""
    if isinstance(as_of, datetime):
        return ensure_utc(as_of)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        snapshot = (
            (
                await session.execute(
                    select(DatasetSnapshot).where(
                        DatasetSnapshot.dataset_id == dataset_id,
                        DatasetSnapshot.name == as_of,
                    )
                )
            )
            .scalars()
            .first()
        )

    if snapshot is None:
        raise ValueError(f"No snapshot named '{as_of}' exists for dataset {dataset_id}.")

    return ensure_utc(snapshot.as_of_time)


async def _completed_runs(dataset_id: UUID) -> List[Tuple[UUID, datetime]]:
    """(pipeline_run_id, completed_at) for the dataset, deduped, oldest first."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(PipelineRun.pipeline_run_id, PipelineRun.created_at)
                .where(
                    PipelineRun.dataset_id == dataset_id,
                    PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                )
                .order_by(PipelineRun.created_at.asc())
            )
        ).all()

    seen: dict[UUID, datetime] = {}
    for run_id, completed_at in rows:
        # A run id normally completes once; keep the first completion time.
        if run_id not in seen:
            seen[run_id] = ensure_utc(completed_at)
    return list(seen.items())


async def get_allowed_run_ids(dataset_id: UUID, as_of_time: datetime) -> set:
    """Run ids completed at or before ``as_of_time`` — the visible history at T."""
    as_of_time = ensure_utc(as_of_time)
    return {
        str(run_id)
        for run_id, completed_at in await _completed_runs(dataset_id)
        if completed_at <= as_of_time
    }


async def get_run_ids_after(dataset_id: UUID, as_of_time: datetime) -> List[str]:
    """Run ids completed after ``as_of_time``, newest first (rollback order)."""
    as_of_time = ensure_utc(as_of_time)
    later = [
        (run_id, completed_at)
        for run_id, completed_at in await _completed_runs(dataset_id)
        if completed_at > as_of_time
    ]
    later.sort(key=lambda item: item[1], reverse=True)
    return [str(run_id) for run_id, _completed_at in later]


async def get_latest_completed_run_id(
    dataset_id: UUID, as_of_time: Optional[datetime] = None
) -> Optional[UUID]:
    """Newest completed run id at ``as_of_time`` (now when omitted)."""
    cutoff = ensure_utc(as_of_time) if as_of_time else datetime.now(timezone.utc)
    candidates = [
        (run_id, completed_at)
        for run_id, completed_at in await _completed_runs(dataset_id)
        if completed_at <= cutoff
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]
