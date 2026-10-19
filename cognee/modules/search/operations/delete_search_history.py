"""Purge recorded questions and answers.

Two scopes, each its own function so the scope is part of the name: a call
site that means "this user's history" cannot reach for the dataset-wide
delete by mistake. Both run regardless of ``COGNEE_LOG_SEARCH_HISTORY``: rows
written while logging was on must still be removable after it is turned off.
"""

from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Query import Query
from ..models.Result import Result


async def _delete_where(column_name: str, value: UUID) -> int:
    db_engine = get_relational_engine()
    removed = 0

    async with db_engine.get_async_session() as session:
        for model in (Query, Result):
            statement = delete(model).where(getattr(model, column_name) == value)
            result = await session.execute(statement)
            removed += result.rowcount or 0

        await session.commit()

    return removed


async def delete_user_search_history(user_id: UUID) -> int:
    """Delete every row the user owns, whichever dataset it was attributed to.

    Includes rows a multi-dataset search left unattributed. Returns the number
    of rows removed across both tables.
    """
    return await _delete_where("user_id", user_id)


async def delete_dataset_search_history(dataset_id: UUID) -> int:
    """Delete every row attributed to the dataset, whoever asked.

    Deleting a dataset removes it for every user it was shared with, so the
    answers another user got from it must not outlive it. The caller is
    responsible for the ``delete`` permission check on the dataset; nothing
    here scopes by user. Returns the number of rows removed across both tables.
    """
    return await _delete_where("dataset_id", dataset_id)
