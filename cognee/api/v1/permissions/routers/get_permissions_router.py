from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.modules.users.exceptions import UserNotFoundError, GroupNotFoundError
from cognee.modules.users import get_user_db
from cognee.modules.users.models import User, Group, Permission, UserGroup, GroupPermission


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/groups/{group_id}/permissions")
    async def give_permission_to_group(
        group_id: str, permission: str, db: Session = Depends(get_user_db)
    ):
        group = (
            (await db.session.execute(select(Group).where(Group.id == group_id))).scalars().first()
        )

        if not group:
            raise GroupNotFoundError

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
                insert(GroupPermission).values(
                    group_id=group.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Group permission already exists.")

        await db.session.commit()

        return JSONResponse(status_code=200, content={"message": "Permission assigned to group"})

    @permissions_router.post("/users/{user_id}/groups")
    async def add_user_to_group(user_id: str, group_id: str, db: Session = Depends(get_user_db)):
        user = (await db.session.execute(select(User).where(User.id == user_id))).scalars().first()
        group = (
            (await db.session.execute(select(Group).where(Group.id == group_id))).scalars().first()
        )

        if not user:
            raise UserNotFoundError
        elif not group:
            raise GroupNotFoundError

        try:
            # Add association directly to the association table
            stmt = insert(UserGroup).values(user_id=user_id, group_id=group_id)
            await db.session.execute(stmt)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User is already part of group.")

        await db.session.commit()

        return JSONResponse(status_code=200, content={"message": "User added to group"})

    return permissions_router
