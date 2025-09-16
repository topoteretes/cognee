from datetime import datetime, timedelta, timezone
from typing import Any

async def cleanup_unused_data(days_threshold: int, dry_run: bool = True) -> dict[str, Any]:
    """
    Cleanup unused data older than the given days_threshold.

    Args:
        days_threshold (int): The age threshold in days for unused data.
        dry_run (bool): If True, simulate the cleanup without deleting data.

    Returns:
        dict[str, Any]: Summary of the cleanup process.
    """
    if days_threshold < 0:
        raise ValueError("days_threshold cannot be negative")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    # Here, replace with actual data fetching/deletion logic
    unused_data_count = 42  # dummy number

    if dry_run:
        return {
            "status": "dry-run",
            "unused_data_count": unused_data_count,
            "cutoff_date": cutoff.isoformat(),
        }
    else:
        # delete data (dummy simulation)
        deleted_count = unused_data_count
        return {
            "status": "executed",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff.isoformat(),
        }
#upadated code