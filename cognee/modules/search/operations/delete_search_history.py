from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query
from ..models.Result import Result


async def delete_search_history(user_id: UUID, dataset_id: UUID | None = None) -> int:
    """Delete a user's recorded questions and answers.

    With ``dataset_id`` only the rows attributed to that dataset go; without it
    every row the user owns goes, including rows a multi-dataset search left
    unattributed. Returns the number of rows removed across both tables.

    Runs regardless of ``COGNEE_LOG_SEARCH_HISTORY``: rows written while
    logging was on must still be removable after it is turned off.
    """
    db_engine = get_relational_engine()
    removed = 0

    async with db_engine.get_async_session() as session:
        for model in (Query, Result):
            statement = delete(model).where(model.user_id == user_id)
            if dataset_id is not None:
                statement = statement.where(model.dataset_id == dataset_id)
            result = await session.execute(statement)
            removed += result.rowcount or 0

        await session.commit()

    return removed
