from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query
from ..models.Result import Result


async def delete_history(user_id: UUID) -> dict[str, int]:
    """Delete every search/recall history row of ``user_id``: their queries and their results.

    Scoped to the one user by ``user_id`` on both tables; no other user's rows
    are touched. Both deletes run in one transaction, so history is never left
    with results whose query is gone. Returns the row counts.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        results = await session.execute(delete(Result).where(Result.user_id == user_id))
        queries = await session.execute(delete(Query).where(Query.user_id == user_id))
        await session.commit()
    return {"queries_deleted": queries.rowcount or 0, "results_deleted": results.rowcount or 0}
