from uuid import UUID

from sqlalchemy.exc import NoResultFound

from cognee.modules.users.exceptions import CapabilityDeniedError, TenantNotFoundError
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_principal, get_tenant


async def get_capability_scope(
    principal_id: UUID, tenant_id: UUID | None, grant_type: str, requester_id: UUID
) -> UUID:
    """
        Work out which tenant a capability change on this principal targets.

        Derived when the principal carries it: a role knows its tenant, and a
        tenant is its own scope. For a user it is the tenant_id the caller
        gives, or the caller's current tenant when they give none, the same
        tenant create_role acts in. It is never read off the target user,
        because a person can belong to several tenants and users.tenant_id only
        names one of them.

        Nothing here may tell a caller which ids are real. A principal that
        does not exist, a tenant_id that does not exist, and a caller with no
        current tenant all raise the same CapabilityDeniedError the
        authorization check for grant_type raises. A real user id and a made-up
        one therefore answer the same until the caller is authorized.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        tenant_id: Tenant given by the caller. Only read when the principal is
            a user.
        grant_type: Capability the caller checks next, so every failure here
            reads exactly like that check failing.
        requester_id: Id of the user making the request, whose current tenant
            is the default for a user principal.

    Returns:
        UUID: Id of the tenant the capability change is scoped to.

    Raises:
        CapabilityDeniedError: If the principal or the tenant does not exist
            (including a role whose tenant is gone), or the caller gave no
            tenant_id and has no current tenant.
    """
    try:
        principal = await get_principal(principal_id)
    except NoResultFound:
        raise CapabilityDeniedError(grant_type)

    if principal.type == "tenant":
        return principal_id

    if principal.type == "role":
        # get_principal loads the subclass columns, so a role's tenant_id is
        # already on the instance.
        tenant_id = principal.tenant_id
    else:
        if tenant_id is None:
            tenant_id = (await get_user(requester_id)).tenant_id
        if tenant_id is None:
            raise CapabilityDeniedError(grant_type)

    # The authorization check that follows would raise TenantNotFoundError (404)
    # for a tenant that does not exist but CapabilityDeniedError (403) for a
    # foreign one. Checked for roles too: a role's tenant row can be missing on
    # SQLite, which does not enforce the foreign key.
    try:
        await get_tenant(tenant_id)
    except TenantNotFoundError:
        raise CapabilityDeniedError(grant_type)

    return tenant_id
