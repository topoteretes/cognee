from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field

from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend
from cognee.modules.users.authentication.get_api_auth_backend import get_api_auth_backend
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.authentication.token_refresh import (
    create_refresh_token,
    store_refresh_token,
    consume_refresh_token,
    verify_refresh_token_and_get_user,
    revoke_all_refresh_tokens_for_user,
    get_access_token_lifetime_seconds,
)


class RefreshTokenBody(BaseModel):
    """Request body for POST /auth/refresh."""

    refresh_token: str = Field(..., description="The refresh token issued at login.")


def get_auth_router():
    auth_backend = get_client_auth_backend()
    auth_router = get_fastapi_users().get_auth_router(auth_backend)

    @auth_router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)) -> dict:
        return {
            "email": user.email,
        }

    @auth_router.post(
        "/refresh",
        summary="Refresh access token",
        description="Exchange a valid refresh token for a new access token. Optionally rotates the refresh token.",
    )
    async def refresh_tokens(body: RefreshTokenBody) -> dict:
        """Return new access token (and optionally new refresh token) from a valid refresh token."""
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            user = await verify_refresh_token_and_get_user(session, body.refresh_token)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired refresh token",
                )
            # Rotate: invalidate current refresh token and issue a new one
            user_id = user.id
            await consume_refresh_token(session, user_id, body.refresh_token)
            new_refresh_token, expires_at = create_refresh_token(user_id)
            await store_refresh_token(session, user_id, new_refresh_token, expires_at)
        api_backend = get_api_auth_backend()
        strategy = api_backend.get_strategy()
        access_token = await strategy.write_token(user)
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": get_access_token_lifetime_seconds(),
        }

    @auth_router.post(
        "/revoke",
        summary="Revoke all refresh tokens",
        description="Invalidate all refresh tokens for the authenticated user (logout all devices).",
    )
    async def revoke_tokens(user: User = Depends(get_authenticated_user)) -> dict:
        """Revoke all refresh tokens for the current user."""
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            count = await revoke_all_refresh_tokens_for_user(session, user.id)
        return {"revoked": count}

    return auth_router
