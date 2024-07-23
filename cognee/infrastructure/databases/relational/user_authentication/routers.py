from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from cognee.infrastructure.databases.relational.user_authentication.authentication_db import get_user_db, User, Group, Permission

permission_router = APIRouter()

@permission_router.post("/groups/{group_id}/permissions")
async def assign_permission_to_group(group_id: int, permission: str, db: Session = Depends(get_user_db)):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    perm = db.query(Permission).filter(Permission.name == permission).first()
    if not perm:
        perm = Permission(name=permission)
        db.add(perm)
    group.permissions.append(perm)
    db.commit()
    return {"msg": "Permission added to group"}

@permission_router.post("/users/{user_id}/groups")
async def add_user_to_group(user_id: int, group_id: int, db: Session = Depends(get_user_db)):
    user = db.query(User).filter(User.id == user_id).first()
    group = db.query(Group).filter(Group.id == group_id).first()
    if not user or not group:
        raise HTTPException(status_code=404, detail="User or group not found")
    user.groups.append(group)
    db.commit()
    return {"msg": "User added to group"}