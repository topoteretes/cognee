from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook


@with_async_session
async def update_notebook(
    notebook: Notebook,
    session: AsyncSession,
) -> Notebook:
    if notebook not in session:
        session.add(notebook)

    return notebook
