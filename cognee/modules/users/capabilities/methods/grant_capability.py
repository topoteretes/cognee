from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import PrincipalCapability, Role, Tenant


async def grant_capability(
    principal_id: UUID,
    tenant_id: UUID,
    capabilities: list[str] | str,
    granted_by: UUID | None = None,
) -> None:
    """
        Grant one or more capabilities to a principal inside one tenant.

        All of them are written in one transaction, so a batch lands whole or
        not at all.

        Granting is idempotent: a capability the principal already holds leaves
        its row unchanged rather than raising, so callers can retry safely, and
        two concurrent grants of the same capability both succeed. That
        includes granted_by: re-granting does not overwrite who granted it
        first.

        The principal must belong to the tenant the grant is scoped to. A role
        from another tenant or the wrong tenant principal is rejected, because a
        row that violates that would be resolved for the wrong tenant's members.
        Users are not checked against membership here on purpose: an invited
        person exists before they accept, and the resolver's membership gate is
        what keeps an early grant from being effective too early.
    Args:
        principal_id: Id of the principal (user, role or tenant).
        tenant_id: Id of the tenant the capabilities are scoped to.
        capabilities: Names from the CAPABILITY_TYPES catalog. Callers validate
            against the catalog; this function only stores.
        granted_by: Id of the user making the grant, recorded on each new row.
            None when there is no requester, e.g. a grant made through the SDK.

    Raises:
        PermissionDeniedError: If the principal does not belong to the tenant.
    """
    # If only a single capability is provided transform it to a list
    if not isinstance(capabilities, list):
        capabilities = [capabilities]

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

        # dict.fromkeys drops duplicates in the request while keeping its order
        rows = [
            {
                "principal_id": principal_id,
                "tenant_id": tenant_id,
                "capability": capability,
                "granted_by": granted_by,
            }
            for capability in dict.fromkeys(capabilities)
        ]
        if not rows:
            return

        bind = await session.connection()
        insert = sqlite_insert if bind.dialect.name == "sqlite" else pg_insert

        # ON CONFLICT DO NOTHING rather than select-then-insert: two concurrent
        # grants of the same capability would both see it missing, and the
        # second insert would fail on the primary key. Doing nothing on conflict
        # is also what keeps an existing row's granted_by.
        await session.execute(
            insert(PrincipalCapability).on_conflict_do_nothing(
                index_elements=["principal_id", "tenant_id", "capability"]
            ),
            rows,
        )
        await session.commit()
