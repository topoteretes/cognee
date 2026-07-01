from fastapi import Depends, HTTPException, status

from cognee.modules.users.models import User

from .get_authenticated_user import get_authenticated_user


async def get_authenticated_superuser(
    user: User = Depends(get_authenticated_user),
) -> User:
    """Return the authenticated user only when they have superuser privileges."""
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access is required.",
        )

    return user
