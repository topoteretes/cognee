from typing import Union
from uuid import UUID

from cognee.modules.data.exceptions import DatasetTypeError
from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets


async def get_authorized_existing_datasets(
    datasets: Union[list[str], list[UUID], None], permission_type: str, user: User
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
        if all(isinstance(dataset, UUID) for dataset in datasets):
            existing_datasets = await get_specific_user_permission_datasets(
                user.id, permission_type, list(datasets)
            )
        elif all(isinstance(dataset, str) for dataset in datasets):
            # Resolve names against ACL-visible datasets, not only owner-scoped rows.
            # Fixes shared-dataset add/search when a collaborator uses dataset_name=...
            permitted_datasets = await get_all_user_permission_datasets(user, permission_type)
            requested_names = set(datasets)
            existing_datasets = [
                dataset for dataset in permitted_datasets if dataset.name in requested_names
            ]
            resolved_names = {dataset.name for dataset in existing_datasets}
            if resolved_names != requested_names:
                raise PermissionDeniedError(
                    f"Request owner does not have necessary permission: [{permission_type}] "
                    "for all datasets requested."
                )
        else:
            raise DatasetTypeError(
                f"One or more of the provided dataset types is not handled: {datasets}"
            )
    else:
        # If no datasets are provided, work with all existing datasets user has permission for.
        existing_datasets = await get_all_user_permission_datasets(user, permission_type)

    return existing_datasets
