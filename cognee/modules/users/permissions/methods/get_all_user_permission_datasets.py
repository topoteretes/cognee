from cognee.shared.logging_utils import get_logger

from ...models.User import User
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.permissions.methods import get_principal_datasets

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
    # Re-resolve the user within this call so `tenants`/`roles` are eager-loaded
    # against a live session. Callers may pass a User that's detached from its
    # original session — e.g. the asyncio background tasks spawned by
    # remember/cognify/memify(run_in_background=True) — where lazy-loading
    # `awaitable_attrs.tenants` would raise DetachedInstanceError. get_user()
    # selectinloads roles + tenants, so the reads below need no live session.
    from cognee.modules.users.methods.get_user import get_user

    resolved_user = await get_user(user.id) or user

    datasets = list()
    # Get all datasets User has explicit access to
    datasets.extend(await get_principal_datasets(resolved_user, permission_type))

    # Get all tenants user is a part of
    tenants = await resolved_user.awaitable_attrs.tenants
    for tenant in tenants:
        # Get all datasets all tenant members have access to
        datasets.extend(await get_principal_datasets(tenant, permission_type))

        # Get all datasets accessible by roles user is a part of
        roles = await resolved_user.awaitable_attrs.roles
        for role in roles:
            datasets.extend(await get_principal_datasets(role, permission_type))

    # Deduplicate datasets with same ID
    unique = {}
    for dataset in datasets:
        # If the dataset id key already exists, leave the dictionary unchanged.
        unique.setdefault(dataset.id, dataset)

    # Filter out dataset that aren't part of the selected user's tenant
    filtered_datasets = []
    for dataset in list(unique.values()):
        if dataset.tenant_id == resolved_user.tenant_id:
            filtered_datasets.append(dataset)

    return filtered_datasets
