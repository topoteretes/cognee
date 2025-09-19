
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, TypedDict, Literal
from uuid import UUID


class CleanupBase(TypedDict):
    cutoff_date: str
    user_id: str | None


class CleanupDryRun(CleanupBase):
    status: Literal["dry-run"]
    unused_data_count: int


class CleanupExecuted(CleanupBase):
    status: Literal["executed"]
    deleted_count: int


async def cleanup_unused_data(
    days_threshold: int = 30,
    dry_run: bool = False,
    user_id: Optional[UUID] = None,
) -> CleanupDryRun | CleanupExecuted:
    """
    Clean up unused data older than a given number of days.

    Args:
        days_threshold (int): Number of days to consider as threshold.
            Data older than this cutoff_date will be considered unused.
        dry_run (bool): If True, only count unused data without deleting.
        user_id (Optional[UUID]): If provided, restrict cleanup to this user's data.

    Returns:
        dict: Information about the cleanup operation.
    """
    if isinstance(days_threshold, bool) or not isinstance(days_threshold, int) or days_threshold < 0:
        raise ValueError("days_threshold must be an int >= 0")

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    if dry_run:
        unused_data_count = await count_unused(cutoff_date, user_id)
        return {
            "status": "dry-run",
            "unused_data_count": unused_data_count,
            "cutoff_date": cutoff_date.isoformat(),
            "user_id": str(user_id) if user_id is not None else None,
        }

    deleted_count = await delete_unused(cutoff_date, user_id)
    return {
        "status": "executed",
        "deleted_count": deleted_count,
        "cutoff_date": cutoff_date.isoformat(),
        "user_id": str(user_id) if user_id is not None else None,
    }


# --- Supporting stubs (to be replaced with real DB/DAO logic) ---

async def count_unused(cutoff_date: datetime, user_id: Optional[UUID]) -> int:
    # Replace with actual DAO/ORM logic
    return 42


async def delete_unused(cutoff_date: datetime, user_id: Optional[UUID]) -> int:
    # Replace with real batched deletion logic
    return 42
