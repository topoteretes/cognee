from uuid import UUID

from cognee.modules.users.capabilities.methods.get_capability_scope import get_capability_scope
from cognee.modules.users.capabilities.methods.revoke_capability import revoke_capability
from cognee.modules.users.capabilities.methods.validate_capability import validate_capability
from cognee.modules.users.permissions.methods import has_grant_permission
from cognee.modules.users.permissions.permission_types import REVOKE_CAPABILITIES


async def authorized_revoke_capability(
    principal_id: UUID,
    capabilities: list[str] | str,
    requester_id: UUID,
    tenant_id: UUID | None = None,
) -> None:
    """
        Take one or more capabilities away from a principal: a user, a role, or
        a tenant, if the requester is allowed to revoke them.

        A member keeps a capability if another level still grants it, since
        resolution is a union. Revoking something the principal never had
        succeeds and changes nothing, so a retry is safe.

        The requester needs the REVOKE_CAPABILITIES capability in the target
        tenant; the tenant owner always has it.

        A batch is all or nothing: every name is validated before anything is
        removed, and the rows are removed in one statement.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        capabilities: Names from the CAPABILITY_TYPES catalog.
        requester_id: Id of the user making the request.
        tenant_id: Tenant the revoke is scoped to when the principal is a user;
            defaults to the requester's current tenant. Ignored for a role or a
            tenant, whose own tenant is always used.

    Raises:
        CapabilityNotFoundError: If any capability is not in the catalog.
        CapabilityDeniedError: If the requester lacks REVOKE_CAPABILITIES in the
            target tenant, or the principal or tenant does not exist.
    """
    validate_capability(capabilities)
    scope = await get_capability_scope(principal_id, tenant_id, REVOKE_CAPABILITIES, requester_id)
    await has_grant_permission(requester_id, scope, REVOKE_CAPABILITIES)
    await revoke_capability(principal_id, scope, capabilities)
