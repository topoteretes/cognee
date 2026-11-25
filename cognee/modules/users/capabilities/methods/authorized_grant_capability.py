from uuid import UUID

from cognee.modules.users.capabilities.methods.get_capability_scope import get_capability_scope
from cognee.modules.users.capabilities.methods.get_unheld_capabilities import (
    get_unheld_capabilities,
)
from cognee.modules.users.capabilities.methods.grant_capability import grant_capability
from cognee.modules.users.capabilities.methods.validate_capability import validate_capability
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods import has_grant_permission
from cognee.modules.users.permissions.permission_types import GRANT_CAPABILITIES


async def authorized_grant_capability(
    principal_id: UUID,
    capabilities: list[str] | str,
    requester_id: UUID,
    tenant_id: UUID | None = None,
) -> None:
    """
        Grant one or more capabilities to a principal: a user, a role, or a
        whole tenant, if the requester is allowed to make that grant.

        Granting to the tenant reaches every current and future member; granting
        to a role reaches its members; granting to a user reaches that person in
        the given tenant only. The same function serves all three because a
        capability row is (principal, tenant, capability) regardless of what the
        principal is.

        The requester needs the GRANT_CAPABILITIES capability in the target
        tenant, and must hold every capability they grant: GRANT_CAPABILITIES
        lets them pass on what they have, not hand themselves or anyone else
        what they were never given. Assigning a role follows the same rule
        (require_role_capabilities). The tenant owner holds everything, which
        is how the first grant gets made. The requester is recorded as
        granted_by on every new row.

        A batch is all or nothing: every name is validated before anything is
        written, and the rows are written in one transaction. Granting is
        idempotent.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        capabilities: Names from the CAPABILITY_TYPES catalog.
        requester_id: Id of the user making the request.
        tenant_id: Tenant the grant is scoped to when the principal is a user;
            defaults to the requester's current tenant. Ignored for a role or a
            tenant, whose own tenant is always used.

    Raises:
        CapabilityNotFoundError: If any capability is not in the catalog.
        CapabilityDeniedError: If the requester lacks GRANT_CAPABILITIES in the
            target tenant, or the principal or tenant does not exist.
        PermissionDeniedError: If the requester does not hold one of the
            capabilities being granted. The message names all of them.
    """
    validate_capability(capabilities)
    scope = await get_capability_scope(principal_id, tenant_id, GRANT_CAPABILITIES, requester_id)
    await has_grant_permission(requester_id, scope, GRANT_CAPABILITIES)

    # If only a single capability is provided transform it to a list
    if not isinstance(capabilities, list):
        capabilities = [capabilities]

    missing = await get_unheld_capabilities(requester_id, scope, capabilities)
    if missing:
        raise PermissionDeniedError(
            message="User cannot grant capabilities they do not hold: " + ", ".join(missing)
        )

    await grant_capability(principal_id, scope, capabilities, granted_by=requester_id)
