from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook


@with_async_session
async def create_notebook(
    user_id: UUID,
    notebook_name: str,
    session: AsyncSession,
) -> Notebook:
    notebook = Notebook(name=notebook_name, owner_id=user_id, cells=[])

    session.add(notebook)

    await session.commit()

    return notebook
