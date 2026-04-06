from typing import Dict, Optional
from uuid import UUID

from cognee.modules.graph.methods import update_event_status
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user


async def invalidate(
    event_id: str,
    status: str = "cancelled",
    dataset_id: Optional[UUID] = None,
    user: Optional[User] = None,
) -> Dict[str, bool]:
    """Invalidate or update the status of an event in the knowledge graph.

    This is the primary SDK function for agents to correct temporal information
    without deleting the event. For example, marking a planned event as cancelled,
    or a completed event as hypothetical.

    Args:
        event_id: Graph node ID of the event to update.
        status: New status value. One of: completed, planned, in_progress,
                hypothetical, cancelled, unknown. Defaults to "cancelled".
        dataset_id: Dataset containing the event. Required.
        user: User performing the update. Uses default user if None.

    Returns:
        Dict mapping event_id to True if updated, False otherwise.

    Example:
        ```python
        import cognee

        # Mark a planned event as cancelled
        await cognee.invalidate("event-node-id", status="cancelled", dataset_id=ds_id)

        # Correct an event's temporal aspect
        await cognee.invalidate("event-node-id", status="hypothetical", dataset_id=ds_id)
        ```
    """
    if dataset_id is None:
        raise ValueError("dataset_id is required for event invalidation")

    if user is None:
        user = await get_default_user()

    return await update_event_status(
        event_ids=[event_id],
        new_status=status,
        user=user,
        dataset_id=dataset_id,
    )
