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


async def _delete_where(column_name: str, value: UUID) -> dict[str, int]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        queries = await session.execute(delete(Query).where(getattr(Query, column_name) == value))
        results = await session.execute(delete(Result).where(getattr(Result, column_name) == value))
        await session.commit()

    return {"queries_deleted": queries.rowcount or 0, "results_deleted": results.rowcount or 0}


async def delete_user_search_history(user_id: UUID) -> dict[str, int]:
    """Delete every row the user owns, whichever dataset it was recorded for.

    Includes rows saved without a dataset. Returns the rows removed from each
    table: ``{"queries_deleted": n, "results_deleted": m}``.
    """
    return await _delete_where("user_id", user_id)


async def delete_dataset_search_history(dataset_id: UUID) -> dict[str, int]:
    """Delete every row recorded for the dataset, whoever asked.

    Deleting a dataset removes it for every user it was shared with, so the
    answers another user got from it must not outlive it. The caller is
    responsible for the ``delete`` permission check on the dataset; nothing
    here scopes by user. Rows saved without a dataset are not matched.
    Returns the rows removed from each table, as ``delete_user_search_history``.
    """
    return await _delete_where("dataset_id", dataset_id)
