import os
from fastapi import APIRouter, HTTPException
from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.models.User import UserRead, UserCreate


def get_register_router():
    if os.environ.get("COGNEE_PUBLIC_REGISTRATION_ENABLED", "true").lower() not in ["true", "1"]:
        dummy_router = APIRouter()
        @dummy_router.post("/register")
        async def register_disabled():
            raise HTTPException(status_code=403, detail="Public registration is disabled.")
        return dummy_router

    return get_fastapi_users().get_register_router(UserRead, UserCreate)
