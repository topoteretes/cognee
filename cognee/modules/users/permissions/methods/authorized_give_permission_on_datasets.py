from typing import Union, List
from cognee.modules.users.permissions.methods import get_principal
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from uuid import UUID


async def authorized_give_permission_on_datasets(
    principal_id: UUID, dataset_ids: Union[List[UUID], UUID], permission_name: str, user_id: UUID
):
    # TODO: Validate user can give permission to other users for given datasets
    # If only a single dataset UUID is provided transform it to a list
    if not isinstance(dataset_ids, list):
        dataset_ids = [dataset_ids]

    principal = await get_principal(principal_id)

    for dataset_id in dataset_ids:
        await give_permission_on_dataset(principal, dataset_id, permission_name)
