from types import SimpleNamespace

from cognee.shared.logging_utils import get_logger

from ...models.User import User
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.permissions.methods import get_principal_datasets
from cognee.modules.users.permissions.methods import get_role, get_tenant

logger = get_logger()


async def get_all_user_permission_datasets(user: User, permission_type: str) -> list[Dataset]:
    """
        Return a list of datasets the user has permission for.
        If the user is part of a tenant, return datasets his roles have permission for.
    Args:
        user
        permission_type

    Returns:
        list[Dataset]: List of datasets user has permission for
    """
    datasets = list()
    # Get all datasets User has explicit access to
    datasets.extend(await get_principal_datasets(user, permission_type))

    if user.tenant_id:
        # Get all datasets all tenants have access to
        tenant = await get_tenant(user.tenant_id)
        datasets.extend(await get_principal_datasets(tenant, permission_type))

        # Get all datasets Users roles have access to
        if isinstance(user, SimpleNamespace):
            # If simple namespace use roles defined in user
            roles = user.roles
        else:
            roles = await user.awaitable_attrs.roles
        for role in roles:
            datasets.extend(await get_principal_datasets(role, permission_type))

    # Deduplicate datasets with same ID
    unique = {}
    for dataset in datasets:
        # If the dataset id key already exists, leave the dictionary unchanged.
        unique.setdefault(dataset.id, dataset)

    return list(unique.values())
