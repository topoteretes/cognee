from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import PrincipalCapability
from cognee.modules.users.permissions.permission_types import (
    CAPABILITY_TYPES,
    USER_MANAGEMENT_ALLOWED_ROLE_NAMES,
)


async def get_role_capabilities(role_id: UUID, tenant_id: UUID, role_name: str) -> set[str]:
    """
        Return every capability a member receives by being in this role.

        That is the capabilities granted to the role itself. A role named in the
        deprecated USER_MANAGEMENT_ALLOWED_ROLE_NAMES set carries the whole
        catalog instead, because has_grant_permission lets its members pass
        every check whether or not any rows were granted to it.

        Capabilities granted to the tenant are not included: every member holds
        those already, so the role adds nothing on top of them.
    Args:
        role_id: Id of the role.
        tenant_id: Id of the tenant that owns the role.
        role_name: Name of the role, for the deprecated fallback.

    Returns:
        set[str]: Capability names the role carries.
    """
    if role_name in USER_MANAGEMENT_ALLOWED_ROLE_NAMES:
        return set(CAPABILITY_TYPES)

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(PrincipalCapability.capability).where(
                PrincipalCapability.principal_id == role_id,
                PrincipalCapability.tenant_id == tenant_id,
            )
        )
        capabilities = {row[0] for row in result.all()}

    # A name that fell out of the catalog gates nothing, so it must not block
    # an assignment either.
    return capabilities & CAPABILITY_TYPES
