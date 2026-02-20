"""Service for creating, storing, and validating refresh tokens."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from fastapi_users.jwt import generate_jwt, decode_jwt
from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.users.models import User
from cognee.modules.users.models.RefreshToken import RefreshToken
from .config import get_refresh_token_lifetime_seconds

REFRESH_TOKEN_AUDIENCE = ["fastapi-users:refresh"]


def _get_secret() -> str:
    return os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_refresh_token(user_id: UUID) -> tuple[str, datetime]:
    """Create a new refresh JWT and its expiry. Caller must store it via store_refresh_token."""
    lifetime = get_refresh_token_lifetime_seconds()
    expires_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=lifetime)
    payload = {
        "sub": str(user_id),
        "aud": REFRESH_TOKEN_AUDIENCE,
        "type": "refresh",
        "jti": secrets.token_urlsafe(16),
    }
    token = generate_jwt(payload, _get_secret(), lifetime_seconds=lifetime, algorithm="HS256")
    return token, expires_at


async def store_refresh_token(
    session: AsyncSession, user_id: UUID, token: str, expires_at: datetime
) -> None:
    """Persist a refresh token hash for the user."""
    token_hash = _hash_token(token)
    record = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(record)
    await session.commit()


async def verify_refresh_token_and_get_user(session: AsyncSession, token: str) -> User | None:
    """
    Verify the refresh token JWT and that it exists in DB and is not expired.
    Returns the User if valid, None otherwise.
    """
    try:
        data = decode_jwt(token, _get_secret(), REFRESH_TOKEN_AUDIENCE, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if data.get("type") != "refresh":
        return None
    user_id_str = data.get("sub")
    if not user_id_str:
        return None
    try:
        user_id = UUID(user_id_str)
    except (ValueError, TypeError):
        return None
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(RefreshToken).where(
            and_(
                RefreshToken.user_id == user_id,
                RefreshToken.token_hash == token_hash,
                RefreshToken.expires_at > now,
            )
        )
    )
    record = result.scalars().first()
    if not record:
        return None
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalars().first()


async def consume_refresh_token(session: AsyncSession, user_id: UUID, token: str) -> bool:
    """Remove the refresh token from DB (for rotation). Returns True if found and deleted."""
    token_hash = _hash_token(token)
    result = await session.execute(
        delete(RefreshToken).where(
            and_(
                RefreshToken.user_id == user_id,
                RefreshToken.token_hash == token_hash,
            )
        )
    )
    await session.commit()
    return result.rowcount > 0


async def revoke_all_refresh_tokens_for_user(session: AsyncSession, user_id: UUID) -> int:
    """Revoke all refresh tokens for the user. Returns number of tokens revoked."""
    result = await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
    await session.commit()
    return result.rowcount
