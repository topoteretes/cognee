from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import PrincipalCapability, Role, Tenant


async def grant_capability(principal_id: UUID, tenant_id: UUID, capability: str) -> None:
    """
        Grant one capability to a principal inside one tenant.

        Granting is idempotent: a capability the principal already holds leaves
        the row set unchanged rather than raising, so callers can retry safely.

        The principal must belong to the tenant the grant is scoped to. A role
        from another tenant or the wrong tenant principal is rejected, because a
        row that violates that would be resolved for the wrong tenant's members.
        Users are not checked against membership here on purpose: an invited
        person exists before they accept, and the resolver's membership gate is
        what keeps an early grant from being effective too early.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        tenant_id: Id of the tenant the capability is scoped to.
        capability: Name from the CAPABILITY_TYPES catalog. Callers validate
            against the catalog; this function only stores.

    Raises:
        PermissionDeniedError: If the principal does not belong to the tenant.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        role = (
            (await session.execute(select(Role).where(Role.id == principal_id))).scalars().first()
        )
        if role is not None and role.tenant_id != tenant_id:
            raise PermissionDeniedError(message="Principal does not belong to this tenant")

        tenant = (
            (await session.execute(select(Tenant).where(Tenant.id == principal_id)))
            .scalars()
            .first()
        )
        if tenant is not None and tenant.id != tenant_id:
            raise PermissionDeniedError(message="Principal does not belong to this tenant")

        existing = await session.execute(
            select(PrincipalCapability.capability).where(
                PrincipalCapability.principal_id == principal_id,
                PrincipalCapability.tenant_id == tenant_id,
                PrincipalCapability.capability == capability,
            )
        )
        if existing.first() is not None:
            return

        session.add(
            PrincipalCapability(
                principal_id=principal_id,
                tenant_id=tenant_id,
                capability=capability,
            )
        )
        await session.commit()
