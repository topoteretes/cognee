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
        notebook, data_dir = await Notebook.from_ipynb_zip_url(
            zip_url=TUTORIAL_ZIP_URL,
            owner_id=user_id,
            notebook_filename="tutorial.ipynb",
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
            force=force_refresh,
        )

        # Add to session and commit
        session.add(notebook)
        await session.commit()

        # Log data directory location for user reference
        if data_dir:
            print(f"Tutorial data files available at: {data_dir}")
            # You could also store this path in user metadata or notebook metadata

    except Exception as e:
        print(f"Failed to fetch tutorial notebook from {TUTORIAL_ZIP_URL}: {e}")

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
