from typing import Union, List

from cognee.modules.users.permissions.methods import get_principal
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from uuid import UUID


async def authorized_give_permission_on_datasets(
    principal_id: UUID, dataset_ids: Union[List[UUID], UUID], permission_name: str, owner_id: UUID
):
    # If only a single dataset UUID is provided transform it to a list
    if not isinstance(dataset_ids, list):
        dataset_ids = [dataset_ids]

    principal = await get_principal(principal_id)

    # Check if request owner has permission to share dataset access
    datasets = await get_specific_user_permission_datasets(owner_id, "share", dataset_ids)

    # TODO: Do we want to enforce sharing of datasets to only be between users of the same tenant?
    for dataset in datasets:
        await give_permission_on_dataset(principal, dataset.id, permission_name)
