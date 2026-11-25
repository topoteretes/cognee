from uuid import UUID

from cognee.modules.users.exceptions import CapabilityDeniedError
from cognee.modules.users.permissions.methods import has_grant_permission


async def get_unheld_capabilities(
    requester_id: UUID, tenant_id: UUID, capabilities: list[str] | set[str]
) -> list[str]:
    """
        Return the capabilities, out of the given ones, the requester does not
        hold in the tenant.

        "Holds" means passes has_grant_permission for it, so the tenant owner,
        the deprecated admin role and tenant-wide grants count exactly as they
        do everywhere else. Used wherever a requester hands capabilities to
        someone, directly or through a role, so that nobody can give out more
        than they have.
    Args:
        requester_id: Id of the user handing the capabilities out.
        tenant_id: Id of the tenant they are handed out in.
        capabilities: Names from the CAPABILITY_TYPES catalog.

    Returns:
        list[str]: The names the requester does not hold, sorted; empty when
        they hold all of them.
    """
    missing = []
    for capability in sorted(set(capabilities)):
        try:
            await has_grant_permission(requester_id, tenant_id, capability)
        except CapabilityDeniedError:
            missing.append(capability)
    return missing
