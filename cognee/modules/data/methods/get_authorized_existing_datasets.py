from typing import Union
from uuid import UUID

from cognee.modules.data.models import Dataset
from cognee.modules.users.models import User
from cognee.modules.data.methods.get_dataset_ids import get_dataset_ids
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets


async def get_authorized_existing_datasets(
    datasets: Union[list[str], list[UUID]], permission_type: str, user: User
) -> list[Dataset]:
    """
    Function returns a list of existing dataset objects user has access for based on datasets input.

    Args:
        datasets:
        user:

    Returns:
        list of Dataset objects

    """
    if datasets:
        # Function handles transforming dataset input to dataset IDs (if possible)
        dataset_ids = await get_dataset_ids(datasets, user)
        # If dataset_ids are provided filter these datasets based on what user has permission for.
        if dataset_ids:
            existing_datasets = await get_specific_user_permission_datasets(
                user.id, permission_type, dataset_ids
            )
        else:
            existing_datasets = []
    else:
        # If no datasets are provided, work with all existing datasets user has permission for.
        existing_datasets = await get_all_user_permission_datasets(user, permission_type)

    return existing_datasets
