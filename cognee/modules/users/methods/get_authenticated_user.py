from ..get_fastapi_users import get_fastapi_users
from fastapi import Depends, HTTPException, Header
import os
import jwt

fastapi_users = get_fastapi_users()

# get_authenticated_user = fastapi_users.current_user(active=True, verified=True)


async def get_authenticated_user(authorization: str = Header(...)):
    """Extract and validate JWT from Authorization header."""
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")

        payload = jwt.decode(
            token, os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret"), algorithms=["HS256"]
        )
        return {
            "user_id": payload["user_id"],
            "tenant_id": payload["tenant_id"],
            "role": payload["role"],
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
