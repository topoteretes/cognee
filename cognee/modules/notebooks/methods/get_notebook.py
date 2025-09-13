from uuid import UUID
from typing import Optional
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook


@with_async_session
async def get_notebook(
    notebook_id: UUID,
    user_id: UUID,
    session: AsyncSession,
) -> Optional[Notebook]:
    result = await session.execute(
        select(Notebook).where(and_(Notebook.owner_id == user_id, Notebook.id == notebook_id))
    )

    return result.scalar()
