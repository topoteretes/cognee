"""Unit tests for the cross-user guard in cognee.modules.integrations.credentials.

The DB is mocked — the invariant under test is policy, not persistence: a
workspace already active for one user must not be silently reassigned to
another on reconnect, while a same-user reconnect (token refresh) still goes
through.
"""

import base64
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from cognee.modules.integrations.credentials import (
    STATUS_ACTIVE,
    STATUS_REVOKED,
    CrossUserConflictError,
    upsert_credential,
)

PROVIDER = "slack"
ACCOUNT_ID = "T123"
USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture(autouse=True)
def _credentials_key(monkeypatch):
    # encrypt_credentials runs before the guard, so it needs a valid key.
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())


def make_existing(user_id: UUID, status: str = STATUS_ACTIVE) -> MagicMock:
    credential = MagicMock()
    credential.user_id = user_id
    credential.status = status
    return credential


def make_session(existing: MagicMock | None) -> MagicMock:
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = existing
    session = MagicMock()
    session.execute = AsyncMock(return_value=execute_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def make_engine(session: MagicMock) -> MagicMock:
    engine = MagicMock()

    @asynccontextmanager
    async def get_async_session():
        yield session

    engine.get_async_session = get_async_session
    return engine


async def _upsert(user_id: UUID, session: MagicMock):
    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        return await upsert_credential(
            provider=PROVIDER,
            user_id=user_id,
            provider_account_id=ACCOUNT_ID,
            token_payload={"access_token": "xoxb-secret"},
        )


@pytest.mark.asyncio
async def test_different_user_active_is_refused():
    session = make_session(make_existing(USER_A, STATUS_ACTIVE))
    with pytest.raises(CrossUserConflictError):
        await _upsert(USER_B, session)
    # Nothing is written when the reconnect is refused.
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_user_reconnect_is_allowed():
    existing = make_existing(USER_A, STATUS_ACTIVE)
    session = make_session(existing)
    await _upsert(USER_A, session)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_takeover_of_revoked_account_is_allowed():
    # User A disconnected (revoked) — user B may now claim the workspace.
    session = make_session(make_existing(USER_A, STATUS_REVOKED))
    await _upsert(USER_B, session)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_connection_inserts():
    session = make_session(existing=None)
    await _upsert(USER_A, session)
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_provider_metadata_merges_without_touching_token_fields():
    from cognee.modules.integrations.credentials import update_provider_metadata

    existing = make_existing(USER_A, STATUS_ACTIVE)
    existing.provider_metadata = {"bot_user_id": "U1"}
    session = make_session(existing)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        updated = await update_provider_metadata(PROVIDER, ACCOUNT_ID, {"allowed_channel_ids": ["C1"]})

    assert updated.provider_metadata == {"bot_user_id": "U1", "allowed_channel_ids": ["C1"]}
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_provider_metadata_returns_none_for_unknown_account():
    from cognee.modules.integrations.credentials import update_provider_metadata

    session = make_session(existing=None)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        updated = await update_provider_metadata(PROVIDER, "unknown-account", {"x": 1})

    assert updated is None
    session.commit.assert_not_awaited()
