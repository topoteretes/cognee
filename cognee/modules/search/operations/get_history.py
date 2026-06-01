from typing import Any, Optional
from uuid import UUID

from sqlalchemy import literal, null, select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query
from ..models.Result import Result


async def get_history(
    user_id: UUID, limit: int = 10, session_id: Optional[str] = None
) -> list[dict[str, Any]]:
    db_engine = get_relational_engine()

    queries_query = select(
        Query.id,
        Query.text.label("text"),
        Query.created_at,
        literal("user").label("user"),
        null().label("query_id"),
    ).filter(Query.user_id == user_id)

    results_query = select(
        Result.id,
        Result.value.label("text"),
        Result.created_at,
        literal("system").label("user"),
        Result.query_id,
    ).filter(Result.user_id == user_id)

    if session_id is not None:
        queries_query = queries_query.filter(Query.session_id == session_id)
        results_query = results_query.filter(
            Result.query_id.in_(
                select(Query.id).filter(Query.user_id == user_id, Query.session_id == session_id)
            )
        )

    history_query = queries_query.union(results_query).order_by("created_at")

    if limit > 0:
        history_query = history_query.limit(limit)

    async with db_engine.get_async_session() as session:
        history = (await session.execute(history_query)).all()

        return [row._asdict() for row in history]
