from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Result import Result
from .search_history_config import search_history_enabled


async def log_result(
    query_id: UUID,
    result: str,
    user_id: UUID,
    dataset_id: UUID | None = None,
):
    if not search_history_enabled():
        return

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(
            Result(
                value=result,
                query_id=query_id,
                user_id=user_id,
                dataset_id=dataset_id,
            )
        )

        await session.commit()
