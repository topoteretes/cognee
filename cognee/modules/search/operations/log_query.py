import os
from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Query import Query

_LOG_ENABLED = os.getenv("COGNEE_LOG_SEARCH_HISTORY", "true").lower() in ("true", "1", "yes")


async def log_query(query_text: str, query_type: str, user_id: UUID) -> Query:
    query = Query(
        text=query_text,
        query_type=query_type,
        user_id=user_id,
    )

    if not _LOG_ENABLED:
        return query

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(query)
        await session.commit()
        return query
