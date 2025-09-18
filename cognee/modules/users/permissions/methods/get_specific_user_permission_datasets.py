from uuid import UUID
from typing import Optional
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.permissions.methods.get_all_user_permission_datasets import (
    get_all_user_permission_datasets,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_user


async def get_specific_user_permission_datasets(
    user_id: UUID, permission_type: str, dataset_ids: Optional[list[UUID]] = None
) -> list[Dataset]:
    """
        Return a list of datasets user has given permission for. If a list of datasets is provided,
        verify for which datasets user has appropriate permission for and return list of datasets he has permission for.
    Args:
        user_id: Id of the user.
        permission_type: Type of the permission.
        dataset_ids: Ids of the provided datasets

    Returns:
        list[Dataset]: List of datasets user has permission for
    """
    user = await get_user(user_id)
    # Find all datasets user has permission for
    user_permission_access_datasets = await get_all_user_permission_datasets(user, permission_type)

    # if specific datasets are provided filter out non provided datasets
    if dataset_ids:
        search_datasets = [
            dataset for dataset in user_permission_access_datasets if dataset.id in dataset_ids
        ]
        # If there are requested datasets that user does not have access to raise error
        if len(search_datasets) != len(dataset_ids):
            raise PermissionDeniedError(
                f"Request owner does not have necessary permission: [{permission_type}] for all datasets requested."
            )
    else:
        search_datasets = user_permission_access_datasets

    if len(search_datasets) == 0:
        raise PermissionDeniedError(
            f"Request owner does not have permission: [{permission_type}] for any dataset."
        )

    return search_datasets
