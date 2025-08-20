import os
from typing import Optional
from fastapi import Depends, HTTPException
from ..models import User
from ..get_fastapi_users import get_fastapi_users
from .get_default_user import get_default_user

# Check environment variable to determine authentication requirement
REQUIRE_AUTHENTICATION = os.getenv("REQUIRE_AUTHENTICATION", "false").lower() == "true"

fastapi_users = get_fastapi_users()

if REQUIRE_AUTHENTICATION:
    # When REQUIRE_AUTHENTICATION=true, enforce authentication (original behavior)
    _auth_dependency = fastapi_users.current_user(active=True)
else:
    # When REQUIRE_AUTHENTICATION=false (default), make authentication optional
    _auth_dependency = fastapi_users.current_user(
        optional=True,  # Returns None instead of raising HTTPException(401)
        active=True,  # Still require users to be active when authenticated
    )


async def get_conditional_authenticated_user(
    user: Optional[User] = Depends(_auth_dependency),
) -> User:
    """
    Get authenticated user with environment-controlled behavior:
    - If REQUIRE_AUTHENTICATION=true: Enforces authentication (raises 401 if not authenticated)
    - If REQUIRE_AUTHENTICATION=false: Falls back to default user if not authenticated

    Always returns a User object for consistent typing.
    """
    if user is None and not REQUIRE_AUTHENTICATION:
        # When authentication is optional and user is None, use default user
        try:
            user = await get_default_user()
        except Exception as e:
            # Convert any get_default_user failure into a proper HTTP 500 error
            raise HTTPException(status_code=500, detail=f"Failed to create default user: {str(e)}")

    return user
