from uuid import UUID
from sqlalchemy import literal, select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Query import Query
from ..models.Result import Result


async def get_history(user_id: UUID, limit: int = 10) -> list[Result]:
    db_engine = get_relational_engine()

    queries_query = select(
        Query.id, Query.text.label("text"), Query.created_at, literal("user").label("user")
    ).filter(Query.user_id == user_id)

    results_query = select(
        Result.id, Result.value.label("text"), Result.created_at, literal("system").label("user")
    ).filter(Result.user_id == user_id)

    history_query = queries_query.union(results_query).order_by("created_at")

    if limit > 0:
        history_query = history_query.limit(limit)

    async with db_engine.get_async_session() as session:
        history = (await session.execute(history_query)).all()

        return history
