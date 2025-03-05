from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.modules.users.exceptions import UserNotFoundError, RoleNotFoundError
from cognee.modules.users import get_user_db
from cognee.modules.users.models import User, Permission, Role, RolePermission, UserRole


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/roles/{role_id}/permissions")
    async def give_permission_to_group(
        role_id: str, permission: str, db: Session = Depends(get_user_db)
    ):
        role = (await db.session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise RoleNotFoundError

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
            # add permission to group
            await db.session.execute(
                insert(RolePermission).values(role_id=role.id, permission_id=permission_entity.id)
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Group permission already exists.")

        await db.session.commit()

        return JSONResponse(status_code=200, content={"message": "Permission assigned to group"})

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

    return permissions_router
