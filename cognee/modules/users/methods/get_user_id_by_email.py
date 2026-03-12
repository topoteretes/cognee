from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import User


async def get_user_id_by_email(user_email: str) -> UUID | None:
    """Return the user ID for a given email, or None if no such user exists."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(select(User.id).where(User.email == user_email))
        return result.scalar()
