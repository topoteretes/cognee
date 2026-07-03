"""Named snapshots: labels pointing at a cut in the run ledger. Zero copies."""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.versioning.methods.timeline import get_latest_completed_run_id
from cognee.modules.versioning.models import DatasetSnapshot


async def create_snapshot(
    dataset_id: UUID, name: str, as_of_time: Optional[datetime] = None
) -> DatasetSnapshot:
    """Label the dataset's current (or given) ledger position with ``name``."""
    if not name or not name.strip():
        raise ValueError("Snapshot name must be a non-empty string.")

    cut = as_of_time or datetime.now(timezone.utc)
    latest_run_id = await get_latest_completed_run_id(dataset_id, cut)

    snapshot = DatasetSnapshot(
        name=name.strip(),
        dataset_id=dataset_id,
        as_of_time=cut,
        latest_pipeline_run_id=latest_run_id,
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        existing = (
            (
                await session.execute(
                    select(DatasetSnapshot).where(
                        DatasetSnapshot.dataset_id == dataset_id,
                        DatasetSnapshot.name == snapshot.name,
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            raise ValueError(f"Snapshot '{snapshot.name}' already exists for dataset {dataset_id}.")
        session.add(snapshot)
        await session.commit()
        # Detach a plain copy so callers can read attributes post-session.
        session.expunge(snapshot)

    return snapshot


async def list_snapshots(dataset_id: UUID) -> List[DatasetSnapshot]:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return list(
            (
                await session.execute(
                    select(DatasetSnapshot)
                    .where(DatasetSnapshot.dataset_id == dataset_id)
                    .order_by(DatasetSnapshot.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
