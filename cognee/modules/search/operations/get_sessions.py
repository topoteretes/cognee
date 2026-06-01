from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query


async def get_sessions(user_id: UUID) -> list[dict[str, Any]]:
    db_engine = get_relational_engine()

    # Subquery: for each session, find the text of the earliest query (by created_at)
    first_query_subq = (
        select(Query.session_id, Query.text)
        .where(Query.user_id == user_id, Query.session_id.isnot(None))
        .order_by(Query.session_id, Query.created_at.asc())
        .distinct(Query.session_id)
        .subquery()
    )

    sessions_query = (
        select(
            Query.session_id,
            first_query_subq.c.text.label("title"),
            func.min(Query.created_at).label("created_at"),
            func.count(Query.id).label("query_count"),
        )
        .join(first_query_subq, Query.session_id == first_query_subq.c.session_id)
        .where(Query.user_id == user_id, Query.session_id.isnot(None))
        .group_by(Query.session_id, first_query_subq.c.text)
        .order_by(func.min(Query.created_at).desc())
    )

    async with db_engine.get_async_session() as session:
        rows = (await session.execute(sessions_query)).all()
        return [row._asdict() for row in rows]
