from typing import Optional
from uuid import UUID
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from ..models import Dataset


async def get_authorized_dataset(
    user_id: UUID, dataset_id: UUID, permission_type="read"
) -> Optional[Dataset]:
    """
    Get a specific dataset with permissions for a user.

    Args:
        user_id (UUID): user id
        dataset_id (UUID): dataset id
        permission_type (str): permission type(read, write, delete, share), default is read

    Returns:
        Optional[Dataset]: dataset with permissions
    """
    datasets = await get_specific_user_permission_datasets(user_id, permission_type, [dataset_id])

    return datasets[0] if datasets else None
