from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import insert

from cognee.modules.users.exceptions import UserNotFoundError, GroupNotFoundError
from cognee.modules.users import get_user_db
from cognee.modules.users.models import User, Group, Permission, UserGroup

def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/groups/{group_id}/permissions")
    async def give_permission_to_group(group_id: str, permission: str, db: Session = Depends(get_user_db)):
        group = db.query(Group).filter(Group.id == group_id).first()

        if not group:
            raise GroupNotFoundError

        permission = db.query(Permission).filter(Permission.name == permission).first()

        if not permission:
            permission = Permission(name = permission)
            db.add(permission)

        group.permissions.append(permission)

        db.commit()

        return JSONResponse(status_code = 200, content = {"message": "Permission assigned to group"})

    @permissions_router.post("/users/{user_id}/groups")
    async def add_user_to_group(user_id: str, group_id: str, db: Session = Depends(get_user_db)):
        user = (await db.session.execute(select(User).where(User.id == user_id))).scalars().first()
        group = (await db.session.execute(select(Group).where(Group.id == group_id))).scalars().first()

        if not user:
            raise UserNotFoundError
        elif not group:
            raise GroupNotFoundError

        # Add association directly to the association table
        stmt = insert(UserGroup).values(user_id=user_id, group_id=group_id)
        await db.session.execute(stmt)
        #user.groups.append(group)

        await db.session.commit()

        return JSONResponse(status_code = 200, content = {"message": "User added to group"})

    return permissions_router
