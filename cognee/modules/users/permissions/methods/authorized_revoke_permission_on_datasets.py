from typing import Union, List
from uuid import UUID

from cognee.modules.users.permissions.methods import (
    get_principal,
    get_specific_user_permission_datasets,
)
from cognee.modules.users.permissions.methods.revoke_permission_on_dataset import (
    revoke_permission_on_dataset,
)


async def authorized_revoke_permission_on_datasets(
    principal_id: UUID, dataset_ids: Union[List[UUID], UUID], permission_name: str, owner_id: UUID
):
    """
    Revoke permission on datasets from a principal.
    The request owner must have share permission on the datasets.

    Args:
        principal_id: Id of the principal whose permission is revoked
        dataset_ids: Ids of datasets to revoke permission on
        permission_name: Name of permission to revoke
        owner_id: Id of the request owner
    """
    if not isinstance(dataset_ids, list):
        dataset_ids = [dataset_ids]

    principal = await get_principal(principal_id)

    datasets = await get_specific_user_permission_datasets(owner_id, "share", dataset_ids)

    for dataset in datasets:
        await revoke_permission_on_dataset(principal, dataset.id, permission_name)
