from uuid import NAMESPACE_OID, UUID, uuid5
from typing import List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook
from .create_tutorial_notebooks import create_tutorial_notebooks

from cognee.shared.logging_utils import get_logger

logger = get_logger()


@with_async_session
async def get_notebooks(
    user_id: UUID,
    session: AsyncSession,
) -> List[Notebook]:
    # Check if tutorial notebook already exists for this user
    tutorial_notebook_ids = [
        uuid5(NAMESPACE_OID, name="Cognee Basics - tutorial 🧠"),
        uuid5(NAMESPACE_OID, name="Python Development with Cognee - tutorial 🧠"),
    ]
    tutorial_query = select(Notebook).where(
        and_(
            Notebook.owner_id == user_id,
            Notebook.id.in_(tutorial_notebook_ids),
            ~Notebook.deletable,
        )
    )
    tutorial_result = await session.execute(tutorial_query)
    tutorial_notebooks = tutorial_result.scalars().all()

    # If tutorial notebooks don't exist, create them
    if len(tutorial_notebooks) == 0:
        logger.info(f"Tutorial notebooks not found for user {user_id}, creating them")
        try:
            await create_tutorial_notebooks(user_id, session)
        except Exception as e:
            # Log the error but continue to return existing notebooks
            logger.error(f"Failed to create tutorial notebook for user {user_id}: {e}")

    # Get all notebooks for the user
    result = await session.execute(select(Notebook).where(Notebook.owner_id == user_id))

    return list(result.scalars().all())
