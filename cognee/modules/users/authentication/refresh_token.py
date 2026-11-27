"""Refresh tokens: a browser login the web UI renews without the password.

An access token cannot be extended (``JWT_LIFETIME_SECONDS``, one hour by
default), so a UI session ended with it. Login also issues a refresh token,
kept in an httpOnly cookie, that ``POST /api/v1/auth/refresh`` exchanges for a
new access token.

Only the token's hash is stored, in ``user_refresh_token``, so signing out ends
the login (``delete_refresh_token``). Each row also holds the fingerprint of
the password it was issued under, keyed with ``FASTAPI_USERS_JWT_SECRET`` like
an access token's: a password change ends the login, and so does a new secret,
which is how an operator signs everyone out. With the secret unset it is
generated per process, so a restart ends every login.

The token is not replaced on use: a browser sends several requests at once,
and with a single-use token every refresh but the first would present one that
no longer exists. Each use moves ``expires_at`` forward instead, so the login
lasts until the UI has not been used for ``REFRESH_TOKEN_LIFETIME_SECONDS``
(7 days by default).

It also ends ``REFRESH_TOKEN_MAX_LIFETIME_SECONDS`` after it started (30 days
by default), however often it was used. Without that end a copied token could
be kept alive for good by using it once a week.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User
from cognee.modules.users.models.UserRefreshToken import UserRefreshToken

from .get_auth_secret import get_auth_secret
from .password_bound_jwt_strategy import password_fingerprint


def get_refresh_token_lifetime() -> timedelta:
    """How long a refresh token stays valid after its last use."""
    return timedelta(seconds=int(os.getenv("REFRESH_TOKEN_LIFETIME_SECONDS", "604800")))


def get_refresh_token_max_lifetime() -> timedelta:
    """How long a refresh token stays valid after the login, however often it is used."""
    return timedelta(seconds=int(os.getenv("REFRESH_TOKEN_MAX_LIFETIME_SECONDS", "2592000")))


def _hash_token(token: str) -> str:
    # A fast hash is enough: the token is 256 random bits, so unlike a
    # password it cannot be guessed from its hash.
    return hashlib.sha256(token.encode()).hexdigest()


def _fingerprint_password(hashed_password: str | None) -> str:
    return password_fingerprint(get_auth_secret("FASTAPI_USERS_JWT_SECRET"), hashed_password)


async def create_refresh_token(user: User) -> str:
    """Store a new refresh token for ``user`` and return it.

    Rows that have expired, of any user, are removed in the same transaction:
    nothing else deletes the token of a browser that never comes back. A token
    past its maximum lifetime is no longer extended, so it expires and is
    removed here within one ``REFRESH_TOKEN_LIFETIME_SECONDS``.
    """
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)

    async with get_relational_engine().get_async_session() as session:
        await session.execute(delete(UserRefreshToken).where(UserRefreshToken.expires_at <= now))
        session.add(
            UserRefreshToken(
                user_id=user.id,
                token_hash=_hash_token(token),
                password_fingerprint=_fingerprint_password(user.hashed_password),
                created_at=now,
                expires_at=now + get_refresh_token_lifetime(),
            )
        )
        await session.commit()

    return token


async def redeem_refresh_token(token: str) -> User | None:
    """Return the user a live refresh token belongs to and extend the token's life.

    None for an unknown token, for one not used for
    ``REFRESH_TOKEN_LIFETIME_SECONDS``, and for one issued
    ``REFRESH_TOKEN_MAX_LIFETIME_SECONDS`` ago, however recently it was used.
    A token whose user is inactive, or whose password or token secret changed
    since it was issued, is deleted: like an access token
    (``PasswordBoundJWTStrategy``) it must not outlive either.
    """
    now = datetime.now(timezone.utc)

    async with get_relational_engine().get_async_session() as session:
        refresh_token = (
            await session.execute(
                select(UserRefreshToken).where(
                    UserRefreshToken.token_hash == _hash_token(token),
                    UserRefreshToken.expires_at > now,
                    UserRefreshToken.created_at > now - get_refresh_token_max_lifetime(),
                )
            )
        ).scalar()
        if refresh_token is None:
            return None

        user = (await session.execute(select(User).filter_by(id=refresh_token.user_id))).scalar()
        if (
            user is None
            or not user.is_active
            or not hmac.compare_digest(
                refresh_token.password_fingerprint, _fingerprint_password(user.hashed_password)
            )
        ):
            await session.delete(refresh_token)
            await session.commit()
            return None

        refresh_token.expires_at = now + get_refresh_token_lifetime()
        await session.commit()
        return user


async def delete_refresh_token(token: str) -> None:
    """End the login ``token`` belongs to. An unknown token is not an error."""
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            delete(UserRefreshToken).where(UserRefreshToken.token_hash == _hash_token(token))
        )
        await session.commit()
