from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID


async def cleanup_unused_data(
    days_threshold: int = 30,
    dry_run: bool = True,
    user_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """
    Cleanup unused data older than the threshold.

    Args:
        days_threshold (int): The age threshold in days for unused data. Defaults to 30.
        dry_run (bool): If True, simulate the cleanup without deleting data.
        user_id (Optional[UUID]): If provided, restrict cleanup to this user’s data.

    Returns:
        dict[str, Any]: Summary of the cleanup operation.
    """
    if not isinstance(days_threshold, int) or days_threshold < 0:
        raise ValueError("days_threshold must be an int >= 0")

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    # Here, replace with actual data fetching/deletion logic
    unused_data_count = 42  # dummy number

    if dry_run:
        return {
            "status": "dry-run",
            "unused_data_count": unused_data_count,
            "cutoff_date": cutoff_date.isoformat(),
            "user_id": str(user_id) if user_id is not None else None,
        }

    # delete data (dummy simulation)
    deleted_count = unused_data_count
    return {
        "status": "executed",
        "deleted_count": deleted_count,
        "cutoff_date": cutoff_date.isoformat(),
        "user_id": str(user_id) if user_id is not None else None,
    }
