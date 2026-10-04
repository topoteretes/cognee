"""The server sets the default user's password once, and never changes one it
has (SDK-549).

The SDK and CLI create the default user with no password. A server started
with DEFAULT_USER_PASSWORD gives it one -- but only then. An environment
variable that could rewrite a stored password would be a takeover primitive,
so an existing password is left alone whether it matches or not.
"""

import importlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pwdlib.exceptions import UnknownHashError

from cognee.base_config import get_base_config
from cognee.modules.users.methods.create_default_user import NO_PASSWORD_SENTINEL

# Module object, not the re-exported name: patch.object on the package would
# resolve to the function.
mod = importlib.import_module("cognee.modules.users.methods.set_default_user_password_if_unset")


def _stack(user, matches=True, verify_raises=None):
    user_db = MagicMock()
    user_db.get_by_email = AsyncMock(return_value=user)
    user_db.update = AsyncMock()

    manager = MagicMock()
    if verify_raises is not None:
        manager.password_helper.verify_and_update = MagicMock(side_effect=verify_raises)
    else:
        manager.password_helper.verify_and_update = MagicMock(return_value=(matches, None))
    manager.password_helper.hash = MagicMock(return_value="NEW-HASH")

    @asynccontextmanager
    async def session_cm():
        yield MagicMock()

    @asynccontextmanager
    async def user_db_cm(_session):
        yield user_db

    @asynccontextmanager
    async def manager_cm(_user_db):
        yield manager

    engine = MagicMock()
    engine.get_async_session = session_cm
    return engine, user_db_cm, manager_cm, user_db, manager


@pytest.fixture
def run(monkeypatch, caplog):
    async def _run(password, stored_hash, matches=True, verify_raises=None):
        if password is None:
            monkeypatch.delenv("DEFAULT_USER_PASSWORD", raising=False)
        else:
            monkeypatch.setenv("DEFAULT_USER_PASSWORD", password)
        get_base_config.cache_clear()

        user = None if stored_hash is None else MagicMock(hashed_password=stored_hash)
        engine, user_db_cm, manager_cm, user_db, manager = _stack(
            user, matches=matches, verify_raises=verify_raises
        )
        with (
            patch.object(mod, "get_relational_engine", return_value=engine),
            patch.object(mod, "get_user_db_context", user_db_cm),
            patch.object(mod, "get_user_manager_context", manager_cm),
        ):
            result = await mod.set_default_user_password_if_unset()
        get_base_config.cache_clear()
        return result, user_db, manager

    return _run


@pytest.mark.asyncio
async def test_unset_variable_touches_nothing(run):
    result, user_db, _manager = await run(None, NO_PASSWORD_SENTINEL)

    assert result is None
    user_db.get_by_email.assert_not_awaited()
    user_db.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_password_less_account_gets_the_password(run):
    """The SDK-first case: the SDK created the user with no password, -ui sets one."""
    result, user_db, manager = await run("default_password", NO_PASSWORD_SENTINEL)

    assert result is True
    manager.password_helper.hash.assert_called_once_with("default_password")
    user_db.update.assert_awaited_once()
    _, payload = user_db.update.await_args.args
    assert payload == {"hashed_password": "NEW-HASH"}


@pytest.mark.asyncio
async def test_matching_password_is_left_alone(run):
    """Existing deployment whose row already has this password: no write."""
    result, user_db, _ = await run("default_password", "$argon2id$real-hash", matches=True)

    assert result is False
    user_db.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_password_is_never_overridden(run, caplog):
    """The rule: an env var must not rewrite a password the account already has."""
    result, user_db, _ = await run("something-else", "$argon2id$real-hash", matches=False)

    assert result is False
    user_db.update.assert_not_awaited()
    assert any("never overrides" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_unverifiable_hash_is_not_overridden_either(run):
    """An unknown hash format is treated as 'has a password', not as 'no password'."""
    result, user_db, _ = await run(
        "default_password", "$legacy$unknown", verify_raises=UnknownHashError("$legacy$unknown")
    )

    assert result is False
    user_db.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_user_is_a_noop(run):
    result, user_db, _ = await run("default_password", None)

    assert result is None
    user_db.update.assert_not_awaited()
