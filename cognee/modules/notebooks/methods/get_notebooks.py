from uuid import UUID
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook


@with_async_session
async def get_notebooks(
    user_id: UUID,
    session: AsyncSession,
) -> List[Notebook]:
    result = await session.execute(select(Notebook).where(Notebook.owner_id == user_id))

    return list(result.scalars().all())
