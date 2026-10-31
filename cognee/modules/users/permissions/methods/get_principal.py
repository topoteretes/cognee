from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import with_polymorphic

from cognee.infrastructure.databases.relational import get_relational_engine

from ...models.Principal import Principal


async def get_principal(principal_id: UUID):
    """
        Return information about a principal (user, role or tenant) based on its id.

        The instance is fully loaded before the session closes, so callers can
        read subclass attributes on it afterwards.
    Args:
        principal_id: Id of the principal

    Returns:
        principal: Information about the principal

    """
    db_engine = get_relational_engine()

    # Principal uses joined-table inheritance: `Role.name` / `Role.tenant_id`
    # and the User columns live in child tables, not in `principals`. A plain
    # select(Principal) loads the base row only, so a caller reading
    # `principal.name` or `principal.tenant_id` after this session closes
    # triggers a refresh on a detached instance and gets DetachedInstanceError
    # (#4976) — which is why the permissions endpoints worked for user and
    # tenant principals, whose branches only touch base-table columns, and
    # returned 500 for roles.
    #
    # "*" outer-joins every child table, so the returned instance carries its
    # subclass columns and is safe to read detached. Deliberately loaded in
    # this one query rather than by keeping the session open for the caller:
    # every caller runs its own follow-up queries in its own session, and
    # handing them a live session to work inside would nest sessions and let
    # a request hold two pool connections at once.
    principal_with_subclasses = with_polymorphic(Principal, "*")

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(principal_with_subclasses).where(principal_with_subclasses.id == principal_id)
        )
        principal = result.unique().scalar_one()
        return principal
