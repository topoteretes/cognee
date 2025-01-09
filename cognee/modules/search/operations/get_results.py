from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.Result import Result


async def get_results(user_id: UUID, limit: int = 10) -> list[Result]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        results = (
            await session.scalars(
                select(Result)
                .filter(Result.user_id == user_id)
                .order_by(Result.created_at.desc())
                .limit(limit)
            )
        ).all()

        return results
