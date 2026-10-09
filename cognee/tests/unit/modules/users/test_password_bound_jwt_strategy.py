"""Session tokens are revoked by a password change.

A stock fastapi-users JWT carries only the user id, so a session opened on
another device stayed valid after the password changed, until it expired.
These tests pin the binding: a token is accepted only while the password it was
issued under is still the user's password, for both the cookie and the Bearer
strategy, which must agree because /auth/login returns one token for both.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest
from fastapi_users import exceptions
from fastapi_users.jwt import generate_jwt

from cognee.modules.users.authentication.api_bearer import APIJWTStrategy
from cognee.modules.users.authentication.default import DefaultJWTStrategy
from cognee.modules.users.authentication.password_bound_jwt_strategy import (
    PASSWORD_FINGERPRINT_CLAIM,
)

SECRET = "test-secret-0123456789abcdef0123456789"
AUDIENCE = ["fastapi-users:auth"]


def _user(hashed_password="hash-v1"):
    return SimpleNamespace(id=uuid4(), hashed_password=hashed_password)


def _manager(user):
    """A user manager that resolves exactly ``user`` and nothing else."""

    async def get(user_id):
        if user_id != user.id:
            raise exceptions.UserNotExists()
        return user

    return SimpleNamespace(get=AsyncMock(side_effect=get), parse_id=lambda value: value)


def _manager_by_str(user):
    manager = _manager(user)
    manager.parse_id = lambda value: user.id if value == str(user.id) else value
    return manager


@pytest.fixture(params=[DefaultJWTStrategy, APIJWTStrategy], ids=["cookie", "bearer"])
def strategy(request):
    return request.param(SECRET, lifetime_seconds=3600)


@pytest.mark.asyncio
async def test_token_is_valid_while_password_is_unchanged(strategy):
    user = _user()
    token = await strategy.write_token(user)
    assert await strategy.read_token(token, _manager_by_str(user)) is user


@pytest.mark.asyncio
async def test_password_change_revokes_existing_tokens(strategy):
    user = _user("hash-v1")
    first_device, second_device = [await strategy.write_token(user) for _ in range(2)]

    user.hashed_password = "hash-v2"  # what fastapi-users' update() stores on a password change

    manager = _manager_by_str(user)
    assert await strategy.read_token(first_device, manager) is None
    assert await strategy.read_token(second_device, manager) is None
    # A login after the change issues a token that works again.
    assert await strategy.read_token(await strategy.write_token(user), manager) is user


@pytest.mark.asyncio
async def test_token_without_fingerprint_is_rejected(strategy):
    """A stock token, as issued before this change, cannot be revoked, so it is refused."""
    user = _user()
    legacy = generate_jwt({"sub": str(user.id), "aud": AUDIENCE}, SECRET, 3600)
    assert await strategy.read_token(legacy, _manager_by_str(user)) is None


@pytest.mark.asyncio
async def test_fingerprint_does_not_expose_the_password_hash(strategy):
    user = _user("$argon2id$v=19$m=65536,t=3,p=4$secret-hash")
    token = await strategy.write_token(user)
    claims = jwt.decode(token, options={"verify_signature": False})
    assert user.hashed_password not in token
    assert len(claims[PASSWORD_FINGERPRINT_CLAIM]) == 64
    assert set(claims) == {"sub", "aud", "exp", PASSWORD_FINGERPRINT_CLAIM}


@pytest.mark.asyncio
async def test_forged_or_foreign_tokens_are_rejected(strategy):
    user = _user()
    manager = _manager_by_str(user)
    other_secret = type(strategy)("another-secret-0123456789abcdef0123", lifetime_seconds=3600)
    assert await strategy.read_token(await other_secret.write_token(user), manager) is None
    assert await strategy.read_token("not-a-jwt", manager) is None
    assert await strategy.read_token(None, manager) is None
    stranger = _user()
    assert await strategy.read_token(await strategy.write_token(stranger), manager) is None


@pytest.mark.asyncio
async def test_cookie_and_bearer_strategies_accept_each_others_tokens():
    """/auth/login writes with the cookie strategy and returns the token as Bearer too."""
    user = _user()
    manager = _manager_by_str(user)
    cookie = DefaultJWTStrategy(SECRET, lifetime_seconds=3600)
    bearer = APIJWTStrategy(SECRET, lifetime_seconds=3600)
    assert await bearer.read_token(await cookie.write_token(user), manager) is user
    assert await cookie.read_token(await bearer.write_token(user), manager) is user
