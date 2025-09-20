from uuid import UUID
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session

from ..models.Notebook import Notebook, NotebookCell

TUTORIAL_NOTEBOOK_NAME = "Python Development with Cognee Tutorial 🧠"


async def _create_tutorial_notebook(
    user_id: UUID, session: AsyncSession, force_refresh: bool = False
) -> None:
    """
    Create the default tutorial notebook for new users.
    Dynamically fetches from: https://github.com/topoteretes/cognee/blob/notebook_tutorial/notebooks/starter_tutorial.zip
    """
    TUTORIAL_ZIP_URL = (
        "https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip"
    )

    try:
        # Create notebook from remote zip file (includes notebook + data files)
        notebook = await Notebook.from_ipynb_zip_url(
            zip_url=TUTORIAL_ZIP_URL,
            owner_id=user_id,
            notebook_filename="tutorial.ipynb",
            name=TUTORIAL_NOTEBOOK_NAME,
            deletable=False,
            force=force_refresh,
        )

        # Add to session and commit
        session.add(notebook)
        await session.commit()

    except Exception as e:
        print(f"Failed to fetch tutorial notebook from {TUTORIAL_ZIP_URL}: {e}")

        raise e


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
