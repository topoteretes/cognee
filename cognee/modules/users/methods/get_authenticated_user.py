from types import SimpleNamespace

from ..get_fastapi_users import get_fastapi_users
from fastapi import HTTPException, Header
import os
import jwt

fastapi_users = get_fastapi_users()


async def get_authenticated_user(authorization: str = Header(...)) -> SimpleNamespace:
    """Extract and validate JWT from Authorization header."""
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")

        payload = jwt.decode(
            token, os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret"), algorithms=["HS256"]
        )

        # SimpleNamespace lets us access dictionary elements like attributes
        auth_data = SimpleNamespace(
            id=payload["user_id"], tenant_id=payload["tenant_id"], roles=payload["roles"]
        )
        return auth_data

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
