from uuid import UUID

from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError, RoleNotFoundError
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods.get_principal import get_principal
from cognee.modules.users.permissions.methods.get_principal_datasets import (
    get_principal_datasets,
)
from cognee.modules.users.permissions.methods.get_user_role_names_in_tenant import (
    get_user_role_names_in_tenant,
)
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)

from .get_all_user_permission_datasets import get_all_user_permission_datasets


async def authorized_get_principal_datasets(
    principal_id: UUID, permission_name: str, requester_id: UUID
) -> list[Dataset]:
    """
        Return the datasets a principal holds a permission on, if the requester
        is allowed to ask about that principal.

        A principal is a user, a role or a tenant, and each answers to a
        different rule:

        * user — themselves, or anyone if they can manage users in the tenant.
        * role — a role of this tenant they are a member of, or any role of it
          if they can manage users.
        * tenant — only the tenant they are currently in.

        The tenant is read off the requester rather than taken as an argument:
        it is the tenant they are currently in, and a caller must not be able
        to name a different one. Results are narrowed to it, so a principal
        that spans tenants never discloses another tenant's datasets.
    Args:
        principal_id: Id of the principal whose datasets are requested
        permission_name: Name of the permission to list ("read", "write", ...)
        requester_id: Id of the user making the request

    Returns:
        list[Dataset]: The principal's datasets within the requester's tenant.

    Raises:
        PermissionDeniedError: If the requester may not ask about this principal.
        RoleNotFoundError: If the role is not part of the requester's tenant.
        UserNotFoundError: If the requester does not exist.
    """
    requester = await get_user(requester_id)
    tenant_id = requester.tenant_id

    principal = await get_principal(principal_id)

    if principal.type == "user":
        if principal.id != requester_id:
            await has_user_management_permission(requester_id, tenant_id)
    elif principal.type == "role":
        # Scoped before the membership test so a role id from another tenant is
        # indistinguishable from one that does not exist.
        if principal.tenant_id != tenant_id:
            raise RoleNotFoundError(message="Role not found for this tenant.")
        # Role names are unique per tenant, so matching on name is a membership
        # test. Members may ask about their own roles; everyone else needs the
        # tenant-wide permission, which raises when absent.
        if principal.name not in await get_user_role_names_in_tenant(requester_id, tenant_id):
            await has_user_management_permission(requester_id, tenant_id)
    elif principal.type == "tenant":
        if principal.id != tenant_id:
            raise PermissionDeniedError(
                message="User is not authorized to read datasets of another tenant"
            )
    else:
        raise PermissionDeniedError(message=f"Unsupported principal type: {principal.type}")

    if principal.type == "user" and principal.id == requester_id:
        # The plugin dataset picker needs effective write access, including
        # tenant and role grants, rather than direct ACL rows alone.
        datasets = await get_all_user_permission_datasets(requester, permission_name)
    else:
        datasets = await get_principal_datasets(principal, permission_name)

    return [dataset for dataset in datasets if dataset.tenant_id == tenant_id]
