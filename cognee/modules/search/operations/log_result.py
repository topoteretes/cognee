import os
from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Result import Result

_LOG_ENABLED = os.getenv("COGNEE_LOG_SEARCH_HISTORY", "true").lower() in ("true", "1", "yes")


async def log_result(query_id: UUID, result: str, user_id: UUID):
    if not _LOG_ENABLED:
        return

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(
            Result(
                value=result,
                query_id=query_id,
                user_id=user_id,
            )
        )

        await session.commit()
