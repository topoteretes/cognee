from uuid import UUID, uuid4
from fastapi_users.exceptions import UserAlreadyExists
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.notebooks.models.Notebook import Notebook
from cognee.modules.users.exceptions import TenantNotFoundError
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.models.User import UserCreate
from cognee.modules.users.models.Tenant import Tenant

from sqlalchemy import select
from typing import Optional


async def _create_tutorial_notebook(user_id: UUID, session: AsyncSession) -> None:
    """
    Create the default tutorial notebook for new users.
    Dynamically fetches from: https://github.com/topoteretes/cognee/blob/notebook_tutorial/notebooks/tutorial.ipynb
    """
    TUTORIAL_NOTEBOOK_URL = "https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/notebook_tutorial/notebooks/tutorial.ipynb"

    try:
        # Create notebook from remote Jupyter notebook
        notebook = await Notebook.from_ipynb_url(
            url=TUTORIAL_NOTEBOOK_URL,
            owner_id=user_id,
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
        )

        # Add to session and commit
        session.add(notebook)
        await session.commit()

    except Exception as e:
        print(f"Failed to fetch tutorial notebook from {TUTORIAL_NOTEBOOK_URL}: {e}")

        raise e


async def create_user(
    email: str,
    password: str,
    tenant_id: Optional[str] = None,
    is_superuser: bool = False,
    is_active: bool = True,
    is_verified: bool = False,
    auto_login: bool = False,
):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    if tenant_id:
                        # Check if the tenant already exists
                        result = await session.execute(select(Tenant).where(Tenant.id == tenant_id))
                        tenant = result.scalars().first()
                        if not tenant:
                            raise TenantNotFoundError

                        user = await user_manager.create(
                            UserCreate(
                                email=email,
                                password=password,
                                tenant_id=tenant.id,
                                is_superuser=is_superuser,
                                is_active=is_active,
                                is_verified=is_verified,
                            )
                        )
                    else:
                        user = await user_manager.create(
                            UserCreate(
                                email=email,
                                password=password,
                                is_superuser=is_superuser,
                                is_active=is_active,
                                is_verified=is_verified,
                            )
                        )

                    if auto_login:
                        await session.refresh(user)

                    await _create_tutorial_notebook(user.id, session)

                    return user
    except UserAlreadyExists as error:
        print(f"User {email} already exists")
        raise error
