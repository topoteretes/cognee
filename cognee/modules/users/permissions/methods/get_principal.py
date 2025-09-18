from sqlalchemy import select
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from ...models.Principal import Principal


async def get_principal(principal_id: UUID):
    """
        Return information about a user based on their id
    Args:
        principal_id: Id of the user

    Returns:
        principal: Information about the user (principal)

    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Principal).where(Principal.id == principal_id))
        principal = result.unique().scalar_one()
        return principal
