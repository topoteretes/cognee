from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from ..models import Dataset


async def get_authorized_dataset(
    user: User, dataset_id: UUID, permission_type="read"
) -> Optional[Dataset]:
    """
    Get a specific dataset with permissions for a user.

    Args:
        user: User object
        dataset_id (UUID): dataset id
        permission_type (str): permission type(read, write, delete, share), default is read

    Returns:
        Optional[Dataset]: dataset with permissions
    """
    authorized_datasets = await get_authorized_existing_datasets(
        [dataset_id], permission_type, user
    )

    return authorized_datasets[0] if authorized_datasets else None
