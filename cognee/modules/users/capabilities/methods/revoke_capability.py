from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import PrincipalCapability


async def revoke_capability(
    principal_id: UUID, tenant_id: UUID, capabilities: list[str] | str
) -> None:
    """
        Take one or more capabilities away from a principal inside one tenant.

        All of them are removed in one statement, so a batch lands whole or not
        at all.

        Revoking is idempotent: a capability the principal never held leaves the
        row set unchanged rather than raising, so callers can retry safely. No
        existence checks either: deleting a row that is not there and deleting
        on behalf of a principal that is not there are the same no-op.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        tenant_id: Id of the tenant the capabilities are scoped to.
        capabilities: Names from the CAPABILITY_TYPES catalog.
    """
    # If only a single capability is provided transform it to a list
    if not isinstance(capabilities, list):
        capabilities = [capabilities]

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        await session.execute(
            delete(PrincipalCapability).where(
                PrincipalCapability.principal_id == principal_id,
                PrincipalCapability.tenant_id == tenant_id,
                PrincipalCapability.capability.in_(capabilities),
            )
        )
        await session.commit()
