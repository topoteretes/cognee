from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Query import Query


async def log_query(query_text: str, query_type: str, user_id: UUID) -> Query:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = Query(
            text=query_text,
            query_type=query_type,
            user_id=user_id,
        )

        session.add(query)

        await session.commit()

        return query
