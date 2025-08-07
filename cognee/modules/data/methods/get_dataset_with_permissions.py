from typing import Optional
from uuid import UUID
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from ..models import Dataset


async def get_dataset_with_permissions(
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
    try:
        datasets = await get_specific_user_permission_datasets(
            user_id, permission_type, [dataset_id]
        )
    except PermissionDeniedError:
        return None

    return datasets[0]
