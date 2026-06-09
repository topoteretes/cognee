from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query


async def get_conversations(user_id: UUID) -> list[dict[str, Any]]:
    db_engine = get_relational_engine()

    conversations_query = (
        select(
            Query.conversation_id,
            func.min(Query.text).label("title"),
            func.min(Query.created_at).label("created_at"),
            func.count(Query.id).label("query_count"),
        )
        .filter(Query.user_id == user_id, Query.conversation_id.isnot(None))
        .group_by(Query.conversation_id)
        .order_by(func.min(Query.created_at).desc())
    )

    async with db_engine.get_async_session() as session:
        rows = (await session.execute(conversations_query)).all()
        return [row._asdict() for row in rows]
