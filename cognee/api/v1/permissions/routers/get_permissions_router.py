from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    TenantNotFoundError,
)
from cognee.modules.users import get_user_db, get_async_session
from cognee.modules.users.models import (
    User,
    Permission,
    Role,
    RoleDefaultPermission,
    UserRole,
    Tenant,
    TenantDefaultPermission,
    UserDefaultPermission,
)


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/roles/{role_id}/permissions")
    async def give_default_permission_to_role(
        role_id: str, permission: str, db: Session = Depends(get_async_session)
    ):
        role = (await db.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise RoleNotFoundError

        permission_entity = (
            (await db.execute(select(Permission).where(Permission.name == permission)))
            .scalars()
            .first()
        )

        if not permission_entity:
            stmt = insert(Permission).values(name=permission)
            await db.execute(stmt)
            permission_entity = (
                (await db.execute(select(Permission).where(Permission.name == permission)))
                .scalars()
                .first()
            )

        try:
            # add default permission to role
            await db.execute(
                insert(RoleDefaultPermission).values(
                    role_id=role.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Role permission already exists.")

        await db.commit()

        return JSONResponse(status_code=200, content={"message": "Permission assigned to role"})

    @permissions_router.post("/tenants/{tenant_id}/permissions")
    async def give_default_permission_to_tenant(
        tenant_id: str, permission: str, db: Session = Depends(get_async_session)
    ):
        tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalars().first()

        if not tenant:
            raise TenantNotFoundError

        permission_entity = (
            (await db.execute(select(Permission).where(Permission.name == permission)))
            .scalars()
            .first()
        )

        if not permission_entity:
            stmt = insert(Permission).values(name=permission)
            await db.execute(stmt)
            permission_entity = (
                (await db.execute(select(Permission).where(Permission.name == permission)))
                .scalars()
                .first()
            )

        try:
            # add default permission to tenant
            await db.execute(
                insert(TenantDefaultPermission).values(
                    tenant_id=tenant.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Tenant permission already exists.")

        await db.commit()

        return JSONResponse(status_code=200, content={"message": "Permission assigned to tenant"})

    @permissions_router.post("/users/{user_id}/permissions")
    async def give_default_permission_to_user(
        user_id: str, permission: str, db: Session = Depends(get_user_db)
    ):
        user = (await db.session.execute(select(User).where(User.id == user_id))).scalars().first()

        if not user:
            raise UserNotFoundError

        permission_entity = (
            (await db.session.execute(select(Permission).where(Permission.name == permission)))
            .scalars()
            .first()
        )

        if not permission_entity:
            stmt = insert(Permission).values(name=permission)
            await db.session.execute(stmt)
            permission_entity = (
                (await db.session.execute(select(Permission).where(Permission.name == permission)))
                .scalars()
                .first()
            )

        try:
            # add default permission to user
            await db.session.execute(
                insert(UserDefaultPermission).values(
                    user_id=user.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User permission already exists.")

        await db.session.commit()

        return JSONResponse(status_code=200, content={"message": "Permission assigned to user"})

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(user_id: str, role_id: str, db: Session = Depends(get_user_db)):
        user = (await db.session.execute(select(User).where(User.id == user_id))).scalars().first()
        role = (await db.session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not user:
            raise UserNotFoundError
        elif not role:
            raise RoleNotFoundError

        try:
            # Add association directly to the association table
            stmt = insert(UserRole).values(user_id=user_id, role_id=role_id)
            await db.session.execute(stmt)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User is already part of group.")

        await db.session.commit()

        return JSONResponse(status_code=200, content={"message": "User added to group"})

    @permissions_router.post("/roles")
    async def create_role(
        role_name: str,
        tenant_id: str,
        permissions: Optional[list[str]] = None,
        db: Session = Depends(get_async_session),
    ):
        try:
            # Add association directly to the association table
            role = Role(name=role_name, tenant_id=tenant_id)
            db.add(role)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Role already exists for tenant.")

        await db.commit()
        await db.refresh(role)

        role = (
            (
                await db.execute(
                    select(Role).where(Role.name == role_name).where(Tenant.id == tenant_id)
                )
            )
            .scalars()
            .first()
        )

        if permissions:
            for permission in permissions:
                await give_default_permission_to_role(role_id=role.id, permission=permission, db=db)

        return JSONResponse(status_code=200, content={"message": "Role created for tenant"})

    @permissions_router.post("/tenants")
    async def create_tenant(
        tenant_name: str,
        permissions: Optional[list[str]] = None,
        db: Session = Depends(get_async_session),
    ):
        try:
            # Add association directly to the association table
            tenant = Tenant(name=tenant_name)
            db.add(tenant)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Tenant already exists.")

        await db.commit()
        await db.refresh(tenant)

        tenant = (
            (await db.execute(select(Tenant).where(Tenant.name == tenant_name))).scalars().first()
        )

        if permissions:
            for permission in permissions:
                await give_default_permission_to_tenant(
                    tenant_id=tenant.id, permission=permission, db=db
                )

        return JSONResponse(status_code=200, content={"message": "Tenant created."})

    return permissions_router
