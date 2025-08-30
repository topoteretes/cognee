from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook


@with_async_session
async def delete_notebook(
    notebook: Notebook,
    session: AsyncSession,
) -> None:
    await session.delete(notebook)
