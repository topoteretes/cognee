"""A browser login is renewed with a refresh token, and only while it should be.

An access token lasts an hour and cannot be extended, so a web UI session ended
with it. Login now also issues a refresh token that POST /api/v1/auth/refresh
exchanges for a new access token. The contract: a token names its user and each
use restarts its lifetime, up to a maximum counted from the login; an expired
or deleted token, a changed password or token secret and an inactive user all
end the login. The stored tokens run against a real
database, because the point is what a stored row does; the routes run with the
storage replaced, because the point there is the cookies and status codes.
"""

import asyncio
import hashlib
import importlib
import os
import pathlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import cognee
from cognee.api.v1.users.routers.get_auth_router import get_auth_router
from cognee.modules.users.authentication.refresh_token import (
    create_refresh_token,
    delete_refresh_token,
    get_refresh_token_lifetime,
    get_refresh_token_max_lifetime,
    redeem_refresh_token,
)
from cognee.modules.users.models import User
from cognee.modules.users.models.UserRefreshToken import UserRefreshToken

# The routers package re-exports get_auth_router under the module's name, so look the module up.
ROUTER = importlib.import_module("cognee.api.v1.users.routers.get_auth_router")

_SYSTEM_ROOT = str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent.parent.parent.parent,
            ".cognee_system/test_refresh_token",
        )
    ).resolve()
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    """Point cognee at a database of its own so these rows never touch dev data."""
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    cognee.config.system_root_directory(_SYSTEM_ROOT)

    # The engine is process-global (@lru_cache), so another module in the same
    # run leaves one cached against its own system root.
    create_relational_engine.cache_clear()

    async def _run():
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


async def _session():
    from cognee.infrastructure.databases.relational import get_relational_engine

    return get_relational_engine().get_async_session()


async def _user() -> User:
    user = User(
        id=uuid4(),
        email=f"{uuid4()}@example.com",
        hashed_password="first-password-hash",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    async with await _session() as session:
        session.add(user)
        await session.commit()
    return user


async def _stored(token: str) -> UserRefreshToken | None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with await _session() as session:
        return (
            await session.execute(select(UserRefreshToken).filter_by(token_hash=token_hash))
        ).scalar()


async def _change(model, row_id, **values) -> None:
    async with await _session() as session:
        row = await session.get(model, row_id)
        for name, value in values.items():
            setattr(row, name, value)
        await session.commit()


def _utc(value: datetime) -> datetime:
    # SQLite hands back naive datetimes; stored values are UTC.
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_token_names_its_user_and_each_use_restarts_its_lifetime():
    """A login about to run out is extended by the full lifetime when it is used."""
    user = await _user()
    token = await create_refresh_token(user)
    now = datetime.now(timezone.utc)
    await _change(
        UserRefreshToken, (await _stored(token)).id, expires_at=now + timedelta(minutes=1)
    )

    redeemed = await redeem_refresh_token(token)

    assert redeemed is not None and redeemed.id == user.id
    expires_at = _utc((await _stored(token)).expires_at)
    assert expires_at > now + timedelta(days=6), "the use moved the expiry a lifetime ahead"


@pytest.mark.asyncio
async def test_the_token_is_stored_as_its_hash_only():
    """Reading the table must not hand out a usable token."""
    token = await create_refresh_token(await _user())

    async with await _session() as session:
        rows = (await session.execute(select(UserRefreshToken))).scalars().all()

    assert rows and all(token not in (row.token_hash, row.password_fingerprint) for row in rows)
    assert await _stored(token) is not None


@pytest.mark.asyncio
async def test_an_expired_token_is_refused():
    """Past its lifetime a token no longer names anyone, and using it does not revive it."""
    token = await create_refresh_token(await _user())
    row_id = (await _stored(token)).id
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    await _change(UserRefreshToken, row_id, expires_at=expired)

    assert await redeem_refresh_token(token) is None
    assert _utc((await _stored(token)).expires_at) == expired


@pytest.mark.asyncio
async def test_a_login_ends_after_its_maximum_lifetime_however_often_it_is_used():
    """Each use restarts the idle time, so without this end a copied token never dies."""
    user = await _user()
    token = await create_refresh_token(user)
    row_id = (await _stored(token)).id
    now = datetime.now(timezone.utc)
    used_a_minute_ago = now + timedelta(days=7) - timedelta(minutes=1)

    await _change(
        UserRefreshToken, row_id, created_at=now - timedelta(days=29), expires_at=used_a_minute_ago
    )
    assert (await redeem_refresh_token(token)).id == user.id, "a 29 day old login is renewed"

    await _change(
        UserRefreshToken, row_id, created_at=now - timedelta(days=31), expires_at=used_a_minute_ago
    )
    assert await redeem_refresh_token(token) is None
    assert _utc((await _stored(token)).expires_at) == used_a_minute_ago, "and is not extended"


@pytest.mark.asyncio
async def test_the_login_start_is_kept_when_the_token_is_used():
    """The maximum lifetime counts from the login, not from the last use."""
    token = await create_refresh_token(await _user())
    started = _utc((await _stored(token)).created_at)
    assert datetime.now(timezone.utc) - started < timedelta(minutes=1)

    await redeem_refresh_token(token)

    assert _utc((await _stored(token)).created_at) == started


@pytest.mark.asyncio
async def test_an_unknown_token_is_refused():
    assert await redeem_refresh_token("never-issued") is None
    assert await redeem_refresh_token("") is None


@pytest.mark.asyncio
async def test_a_password_change_ends_the_login():
    """Like an access token, a refresh token must not outlive the password it was issued under."""
    user = await _user()
    token = await create_refresh_token(user)
    await _change(User, user.id, hashed_password="second-password-hash")

    assert await redeem_refresh_token(token) is None
    assert await _stored(token) is None, "the dead token is removed, not left to expire"


@pytest.mark.asyncio
async def test_a_new_token_secret_ends_the_login(monkeypatch):
    """Changing FASTAPI_USERS_JWT_SECRET is how an operator signs everyone out.

    It already ends every access token. A refresh token that survived it would
    hand the browser a new one signed with the new secret.
    """
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "the-secret-the-login-was-opened-under")
    user = await _user()
    token = await create_refresh_token(user)
    assert (await redeem_refresh_token(token)).id == user.id

    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "the-secret-after-a-rotation")

    assert await redeem_refresh_token(token) is None
    assert await _stored(token) is None


@pytest.mark.asyncio
async def test_the_password_fingerprint_is_the_one_access_tokens_carry():
    """One fingerprint for both kinds of token, so they cannot drift apart."""
    from cognee.modules.users.authentication.get_client_auth_backend import (
        get_client_auth_backend,
    )

    user = await _user()
    token = await create_refresh_token(user)

    strategy = get_client_auth_backend().get_strategy()
    stored = (await _stored(token)).password_fingerprint
    assert stored == strategy._password_fingerprint(user.hashed_password)


@pytest.mark.asyncio
async def test_an_inactive_user_cannot_renew_a_login():
    """Login refuses an inactive user; a login opened before must not get around that."""
    user = await _user()
    token = await create_refresh_token(user)
    await _change(User, user.id, is_active=False)

    assert await redeem_refresh_token(token) is None
    assert await _stored(token) is None


@pytest.mark.asyncio
async def test_deleting_a_token_ends_that_login_and_no_other():
    """Signing out in one browser leaves the same user's other browser logged in."""
    user = await _user()
    signed_out = await create_refresh_token(user)
    other_browser = await create_refresh_token(user)

    await delete_refresh_token(signed_out)

    assert await redeem_refresh_token(signed_out) is None
    assert (await redeem_refresh_token(other_browser)).id == user.id


@pytest.mark.asyncio
async def test_a_sign_out_during_a_renewal_ends_the_login_without_an_error():
    """Another tab signs out after this one's renewal has read the token and before it writes.

    Extending the row that was read failed then (StaleDataError), and the route
    answered 500: the UI showed a backend error instead of the login page.
    """
    token = await create_refresh_token(await _user())
    execute = AsyncSession.execute

    async def execute_then_sign_out(session, statement, *args, **kwargs):
        result = await execute(session, statement, *args, **kwargs)
        # A renewal reads the token, then its user, then writes.
        if statement.is_select and statement.column_descriptions[0]["entity"] is User:
            await delete_refresh_token(token)
        return result

    with patch.object(AsyncSession, "execute", execute_then_sign_out):
        assert await redeem_refresh_token(token) is None

    assert await _stored(token) is None


@pytest.mark.asyncio
async def test_a_login_removes_the_tokens_that_have_expired():
    """Nothing else deletes the token of a browser that never comes back."""
    abandoned = await create_refresh_token(await _user())
    await _change(
        UserRefreshToken,
        (await _stored(abandoned)).id,
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    live = await create_refresh_token(await _user())

    await create_refresh_token(await _user())

    assert await _stored(abandoned) is None
    assert await _stored(live) is not None


def test_the_lifetime_is_seven_days_unless_the_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv("REFRESH_TOKEN_LIFETIME_SECONDS", raising=False)
    assert get_refresh_token_lifetime() == timedelta(days=7)

    monkeypatch.setenv("REFRESH_TOKEN_LIFETIME_SECONDS", "90")
    assert get_refresh_token_lifetime() == timedelta(seconds=90)

    monkeypatch.setenv("REFRESH_TOKEN_LIFETIME_SECONDS", "a week")
    with pytest.raises(ValueError):
        get_refresh_token_lifetime()


def test_the_maximum_lifetime_is_thirty_days_unless_the_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv("REFRESH_TOKEN_MAX_LIFETIME_SECONDS", raising=False)
    assert get_refresh_token_max_lifetime() == timedelta(days=30)

    monkeypatch.setenv("REFRESH_TOKEN_MAX_LIFETIME_SECONDS", "3600")
    assert get_refresh_token_max_lifetime() == timedelta(hours=1)

    monkeypatch.setenv("REFRESH_TOKEN_MAX_LIFETIME_SECONDS", "a month")
    with pytest.raises(ValueError):
        get_refresh_token_max_lifetime()


@pytest.mark.parametrize(
    "read, name",
    [
        (get_refresh_token_lifetime, "REFRESH_TOKEN_LIFETIME_SECONDS"),
        (get_refresh_token_max_lifetime, "REFRESH_TOKEN_MAX_LIFETIME_SECONDS"),
    ],
)
@pytest.mark.parametrize("value", ["0", "-5", "1.5", "a week"])
def test_a_lifetime_that_is_not_a_positive_number_of_seconds_is_an_error(
    monkeypatch, read, name, value
):
    """Zero or less would issue tokens that have already expired.

    Every browser would be logged out after one access token, with no error
    anywhere. The message names the setting: the server reads both at start-up
    and stops there.
    """
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"^{name} must be a positive whole number of seconds"):
        read()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(get_auth_router(), prefix="/api/v1/auth")
    return TestClient(app)


def _access_tokens(token: str):
    strategy = MagicMock(write_token=AsyncMock(return_value=token))
    return patch.object(
        ROUTER,
        "get_client_auth_backend",
        return_value=MagicMock(get_strategy=MagicMock(return_value=strategy)),
    )


def test_refresh_exchanges_the_cookie_for_a_new_access_token():
    """The browser gets a new access cookie, and keeps the refresh cookie for another lifetime."""
    user = MagicMock()
    redeem = AsyncMock(return_value=user)
    client = _client()
    client.cookies.set("auth_token_refresh", "refresh-456")

    with patch.object(ROUTER, "redeem_refresh_token", redeem), _access_tokens("token-789"):
        response = client.post("/api/v1/auth/refresh")

    assert response.status_code == 200, response.text
    assert response.json() == {"access_token": "token-789", "token_type": "bearer"}
    redeem.assert_awaited_once_with("refresh-456")
    access, refresh = response.headers.get_list("set-cookie")
    assert access.startswith("auth_token=token-789;") and "Path=/;" in access
    assert refresh.startswith("auth_token_refresh=refresh-456;") and "Max-Age=604800" in refresh
    assert "Path=/api/v1/auth;" in refresh, "the browser sends it to the auth routes only"


@pytest.mark.parametrize("cookie", [None, "refresh-456"])
def test_refresh_without_a_live_token_is_unauthorized(cookie):
    """No cookie and a token the storage refuses get the same 401 and no cookies."""
    redeem = AsyncMock(return_value=None)
    client = _client()
    if cookie:
        client.cookies.set("auth_token_refresh", cookie)

    with patch.object(ROUTER, "redeem_refresh_token", redeem), _access_tokens("token-789"):
        response = client.post("/api/v1/auth/refresh")

    assert response.status_code == 401
    assert response.json() == {"detail": "REFRESH_BAD_TOKEN"}
    assert "set-cookie" not in response.headers
    redeem.assert_awaited_once_with(cookie or "")


def test_logout_ends_the_login_without_a_valid_access_token():
    """An expired access token must not keep a login alive: logout needs the refresh cookie only."""
    delete = AsyncMock()
    client = _client()
    client.cookies.set("auth_token_refresh", "refresh-456")

    with patch.object(ROUTER, "delete_refresh_token", delete):
        response = client.post("/api/v1/auth/logout")

    assert response.status_code == 200 and response.json() == {}
    delete.assert_awaited_once_with("refresh-456")
    access, refresh = response.headers.get_list("set-cookie")
    assert access.startswith('auth_token=""') and "Max-Age=0" in access
    assert refresh.startswith('auth_token_refresh=""') and "Max-Age=0" in refresh
    assert "Path=/;" in access and "Path=/api/v1/auth;" in refresh


def test_logout_without_a_refresh_cookie_only_clears_the_cookies():
    delete = AsyncMock()

    with patch.object(ROUTER, "delete_refresh_token", delete):
        response = _client().post("/api/v1/auth/logout")

    assert response.status_code == 200
    delete.assert_not_awaited()
    assert len(response.headers.get_list("set-cookie")) == 2
