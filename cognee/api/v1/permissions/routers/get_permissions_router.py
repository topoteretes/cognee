from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from cognee.modules.users import get_user_db
from cognee.modules.users.models import User, Group, Permission

def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/groups/{group_id}/permissions")
    async def give_permission_to_group(group_id: int, permission: str, db: Session = Depends(get_user_db)):
        group = db.query(Group).filter(Group.id == group_id).first()

        if not group:
            raise HTTPException(status_code = 404, detail = "Group not found")

        permission = db.query(Permission).filter(Permission.name == permission).first()

        if not permission:
            permission = Permission(name = permission)
            db.add(permission)

        group.permissions.append(permission)

        db.commit()

        return JSONResponse(status_code = 200, content = {"message": "Permission assigned to group"})

    @permissions_router.post("/users/{user_id}/groups")
    async def add_user_to_group(user_id: int, group_id: int, db: Session = Depends(get_user_db)):
        user = db.query(User).filter(User.id == user_id).first()
        group = db.query(Group).filter(Group.id == group_id).first()

        if not user or not group:
            raise HTTPException(status_code = 404, detail = "User or group not found")

        user.groups.append(group)

        db.commit()

        return JSONResponse(status_code = 200, content = {"message": "User added to group"})

    return permissions_router
