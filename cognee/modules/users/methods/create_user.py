from fastapi_users.exceptions import UserAlreadyExists
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.models.User import UserCreate
from cognee.modules.users.models.Tenant import Tenant
from cognee.modules.users.models.Role import Role

from sqlalchemy import select
from typing import Optional, List


async def create_user(
    email: str,
    password: str,
    tenant_name: Optional[str] = "Default",
    role_names: Optional[List[str]] = None,
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
                    # Check if the tenant already exists
                    result = await session.execute(select(Tenant).where(Tenant.name == tenant_name))
                    tenant = result.scalars().first()
                    if not tenant:
                        tenant = Tenant(name=tenant_name)
                        session.add(tenant)
                        await session.commit()
                        await session.refresh(tenant)

                    # Prepare list for roles
                    roles = []
                    if role_names:
                        for role_name in role_names:
                            result = await session.execute(
                                select(Role).where(
                                    Role.name == role_name, Role.tenant_id == tenant.id
                                )
                            )
                            role = result.scalars().first()
                            if not role:
                                role = Role(name=role_name, tenant=tenant)
                                session.add(role)
                                await session.commit()
                                await session.refresh(role)
                            roles.append(role)

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

                    # Explicitly refresh the user and load the roles relationship
                    await session.commit()
                    await session.refresh(user, ["roles"])

                    # Associate any roles that were provided
                    if roles:
                        user.roles.extend(roles)
                        await session.commit()
                        await session.refresh(user)

                    return user
    except UserAlreadyExists as error:
        print(f"User {email} already exists")
        raise error
