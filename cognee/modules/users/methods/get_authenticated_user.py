import os
from typing import Optional
from fastapi import Depends, HTTPException
from ..models import User
from ..get_fastapi_users import get_fastapi_users
from .get_default_user import get_default_user
from cognee.shared.logging_utils import get_logger


logger = get_logger("get_authenticated_user")

# Check environment variable to determine authentication requirement
REQUIRE_AUTHENTICATION = (
    os.getenv("REQUIRE_AUTHENTICATION", "false").lower() == "true"
    or os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true"
)

fastapi_users = get_fastapi_users()

_auth_dependency = fastapi_users.current_user(active=True, optional=not REQUIRE_AUTHENTICATION)


async def get_authenticated_user(
    user: Optional[User] = Depends(_auth_dependency),
) -> User:
    """
    Get authenticated user with environment-controlled behavior:
    - If REQUIRE_AUTHENTICATION=true: Enforces authentication (raises 401 if not authenticated)
    - If REQUIRE_AUTHENTICATION=false: Falls back to default user if not authenticated

    Always returns a User object for consistent typing.
    """
    if user is None:
        # When authentication is optional and user is None, use default user
        try:
            user = await get_default_user()
        except Exception as e:
            # Convert any get_default_user failure into a proper HTTP 500 error
            logger.error(f"Failed to create default user: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create default user: {str(e)}"
            ) from e

    return user
