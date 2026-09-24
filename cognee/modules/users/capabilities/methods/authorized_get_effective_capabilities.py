from uuid import UUID

from cognee.modules.users.permissions.methods import (
    get_effective_capabilities,
    require_tenant_membership,
)


async def authorized_get_effective_capabilities(requester_id: UUID, tenant_id: UUID) -> list[str]:
    """
        Return the capabilities the requester has in a tenant they belong to.

        Capabilities are tenant-scoped actions (as opposed to dataset
        permissions) and are the union of what the tenant and the requester's
        roles in it grant. The tenant owner holds all of them.

        The requester must belong to the tenant. A tenant they are not a member
        of and one that does not exist both raise PermissionDeniedError, so this
        cannot be used to discover which tenant ids are real.

        Intended for a client to decide which controls to show. It is not an
        authorization boundary on its own: every operation still enforces its
        own capability.
    Args:
        requester_id: Id of the user making the request.
        tenant_id: Id of the tenant the capabilities are scoped to.

    Returns:
        list[str]: Capability names, sorted so the result is stable.

    Raises:
        PermissionDeniedError: If the requester is not a member of the tenant,
            or it does not exist.
    """
    await require_tenant_membership(requester_id, tenant_id)

    capabilities = await get_effective_capabilities(requester_id, tenant_id)

    return sorted(capabilities)
