from uuid import UUID

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import User, UserTenant


async def get_user_id_by_email(user_email: str, requester: User) -> UUID | None:
    """Return the id of the user with this email, or None when ``requester`` may not see them.

    Sharing a dataset by email needs the id of the person to share with; the
    lookup must not double as an instance-wide email oracle, confirming to any
    authenticated caller which addresses exist across tenants. A requester
    always finds themselves. One who belongs to tenants finds the members of
    those tenants (``user_tenants``, the owner included). One who belongs to
    no tenant finds users who belong to none: the single-instance shape where
    tenants are not used and independently registered users share by email.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        requester_tenant_ids = (
            (
                await session.execute(
                    select(UserTenant.tenant_id).where(UserTenant.user_id == requester.id)
                )
            )
            .scalars()
            .all()
        )
        if requester_tenant_ids:
            visible = User.id.in_(
                select(UserTenant.user_id).where(UserTenant.tenant_id.in_(requester_tenant_ids))
            )
        else:
            visible = ~User.id.in_(select(UserTenant.user_id))

        result = await session.execute(
            select(User.id).where(User.email == user_email, or_(User.id == requester.id, visible))
        )
        return result.scalar()
