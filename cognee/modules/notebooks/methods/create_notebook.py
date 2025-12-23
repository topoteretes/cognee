from uuid import UUID
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook, NotebookCell


@with_async_session
async def create_notebook(
    user_id: UUID,
    notebook_name: str,
    cells: Optional[List[NotebookCell]],
    deletable: Optional[bool],
    session: AsyncSession,
) -> Notebook:
    notebook = Notebook(
        name=notebook_name, owner_id=user_id, cells=cells, deletable=deletable or True
    )

    session.add(notebook)

    await session.commit()

    return notebook
