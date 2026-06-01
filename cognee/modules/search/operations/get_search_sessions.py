from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query


async def get_search_sessions(user_id: UUID) -> list[dict[str, Any]]:
    """Return session metadata for all sessions belonging to a user.

    Groups queries by session_id and returns one entry per session containing:
    - session_id: the client-provided session identifier
    - title: text of the first query in the session (by created_at)
    - created_at: timestamp of the first query in the session
    - query_count: total number of queries in the session

    Uses a row_number window function to select the earliest query per session,
    compatible with both PostgreSQL and SQLite.
    """
    db_engine = get_relational_engine()

    # Rank queries within each session by creation time (1 = earliest)
    row_num = (
        func.row_number()
        .over(
            partition_by=Query.session_id,
            order_by=Query.created_at.asc(),
        )
        .label("rn")
    )

    ranked = (
        select(Query.session_id, Query.text, row_num)
        .where(Query.user_id == user_id, Query.session_id.isnot(None))
        .subquery()
    )

    first_query_subq = select(ranked.c.session_id, ranked.c.text).where(ranked.c.rn == 1).subquery()

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
