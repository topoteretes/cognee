from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query
from .search_history_config import search_history_enabled


async def log_query(
    query_text: str,
    query_type: str,
    user_id: UUID,
    dataset_id: UUID | None = None,
) -> Query:
    query = Query(
        text=query_text,
        query_type=query_type,
        user_id=user_id,
        dataset_id=dataset_id,
    )

    if not search_history_enabled():
        return query

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(query)
        await session.commit()
        return query
