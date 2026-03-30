from uuid import UUID
from typing import Optional

from deprecated import deprecated

from cognee.api.v1.datasets import datasets
from cognee.modules.users.models import User


@deprecated(
    reason="cognee.delete is deprecated. Use `datasets.delete_data` instead.", version="0.3.9"
)
async def delete(data_id: UUID, dataset_id: UUID, mode: str = "soft", user: Optional[User] = None):
    """
    Remove a specific data item from a dataset.

    Deprecated since version 0.3.9. Use ``cognee.datasets.delete_data`` instead.

    Args:
        data_id: UUID of the data item to remove.
        dataset_id: UUID of the dataset containing the data item.
        mode: Deletion mode. Defaults to "soft".
        user: User context for authentication and permissions. Uses default user if None.

    Returns:
        dict: Status of the deletion operation, e.g. ``{"status": "success"}``.

    Example:
        ```python
        import cognee

        # Preferred approach (not deprecated)
        await cognee.datasets.delete_data(dataset_id=dataset_id, data_id=data_id)

        # Legacy usage (deprecated since 0.3.9)
        await cognee.delete(data_id=data_id, dataset_id=dataset_id)
        ```
    """
    return await datasets.delete_data(data_id=data_id, dataset_id=dataset_id, mode=mode, user=user)
