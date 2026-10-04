from uuid import UUID

from cognee.modules.users.capabilities.methods.get_role_capabilities import get_role_capabilities
from cognee.modules.users.capabilities.methods.get_unheld_capabilities import (
    get_unheld_capabilities,
)
from cognee.modules.users.exceptions import PermissionDeniedError


async def require_role_capabilities(
    requester_id: UUID, role_id: UUID, tenant_id: UUID, role_name: str
) -> None:
    """
        Refuse to hand out a role that carries capabilities the requester does
        not hold themselves.

        MANAGE_USERS lets a requester assign roles. Without this rule it would
        also let them reach every other capability in the tenant: add
        themselves to a role that holds GRANT_CAPABILITIES, or to a role named
        "admin", which the deprecated fallback treats as holding everything.
        Checking the requester against each capability the role carries closes
        both, whoever the role is being given to. authorized_grant_capability
        applies the same rule to direct grants, through the same
        get_unheld_capabilities.
    Args:
        requester_id: Id of the user assigning the role.
        role_id: Id of the role being assigned.
        tenant_id: Id of the tenant that owns the role.
        role_name: Name of the role, for the deprecated fallback.

    Raises:
        PermissionDeniedError: If the role carries any capability the requester
            does not hold. The message names all of them.
    """
    role_capabilities = await get_role_capabilities(role_id, tenant_id, role_name)
    missing = await get_unheld_capabilities(requester_id, tenant_id, role_capabilities)

    if missing:
        raise PermissionDeniedError(
            message="User cannot assign a role that carries capabilities they do not hold: "
            + ", ".join(missing)
        )
