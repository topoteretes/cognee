from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Query import Query


async def get_queries(user_id: UUID, limit: int) -> list[Query]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        queries = (
            await session.scalars(
                select(Query)
                .filter(Query.user_id == user_id)
                .order_by(Query.created_at.desc())
                .limit(limit)
            )
        ).all()

        return queries
