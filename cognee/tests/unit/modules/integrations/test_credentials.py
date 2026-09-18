"""Unit tests for cognee.modules.integrations.credentials.

The DB is mocked — the invariants under test are policy, not persistence. Two
of them: a workspace already active for one user must not be silently
reassigned to another on reconnect, while a same-user reconnect (token
refresh) still goes through; and a reconnect must not destroy the parts of a
credential that no token response can rebuild.
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
    get_active_credential_for_workspace,
    upsert_credential,
)

PROVIDER = "slack"
ACCOUNT_ID = "T123"
USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
WORKSPACE_A = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
WORKSPACE_B = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


@pytest.fixture(autouse=True)
def _credentials_key(monkeypatch):
    # encrypt_credentials runs before the guard, so it needs a valid key.
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())


def make_existing(
    user_id: UUID,
    status: str = STATUS_ACTIVE,
    workspace_id: UUID | None = None,
    provider_metadata: dict | None = None,
) -> MagicMock:
    credential = MagicMock()
    credential.user_id = user_id
    credential.workspace_id = workspace_id
    credential.status = status
    # A real row holds None or a dict here, never an auto-created attribute:
    # upsert merges into this value, so the mock has to be faithful about it.
    credential.provider_metadata = provider_metadata
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


async def _upsert(user_id: UUID, session: MagicMock, **overrides):
    """Upsert against the mocked session, genuinely omitting what is not passed.

    Omission is its own case: an argument left out must leave the stored value
    alone, which is a different outcome from passing ``None`` explicitly.
    """
    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        return await upsert_credential(
            provider=PROVIDER,
            user_id=user_id,
            provider_account_id=ACCOUNT_ID,
            token_payload={"access_token": "xoxb-secret"},
            **overrides,
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
async def test_takeover_of_revoked_account_inherits_the_previous_metadata():
    # The surprising half of merging, pinned so it cannot change unnoticed:
    # the row is reused rather than replaced, so B starts out carrying the
    # allowlist A left behind. That direction is restrictive (B's workspace
    # begins narrowed, never widened) and B can see and change it on the
    # channels page, which is why it is accepted rather than special-cased.
    existing = make_existing(
        USER_A,
        STATUS_REVOKED,
        provider_metadata={"bot_user_id": "U1", "allowed_channel_ids": ["C1", "C2"]},
    )
    session = make_session(existing)

    await _upsert(USER_B, session, provider_metadata={"bot_user_id": "U2"})

    assert existing.provider_metadata == {
        "bot_user_id": "U2",
        "allowed_channel_ids": ["C1", "C2"],
    }


@pytest.mark.asyncio
async def test_first_connection_inserts():
    session = make_session(existing=None)
    await _upsert(USER_A, session)
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_scoped_different_workspace_active_is_refused():
    # Owned by WORKSPACE_A; a different workspace reconnecting must be
    # refused even though the connecting user differs too — workspace_id is
    # the conflict key here, not user_id.
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    session = make_session(existing)
    with pytest.raises(CrossUserConflictError):
        await _upsert(USER_B, session, workspace_id=WORKSPACE_B)
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_workspace_scoped_same_workspace_different_user_reconnect_is_allowed():
    # Same workspace, a different member of it reconnecting (e.g. someone
    # else on the team clicks "Connect" again) takes the row over — the
    # workspace owns it, not whichever user happened to connect it first.
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    session = make_session(existing)
    await _upsert(USER_B, session, workspace_id=WORKSPACE_A)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_omitting_workspace_id_falls_back_to_user_id_even_if_row_has_one():
    # A caller that doesn't pass workspace_id keeps the original single-user
    # contract literally: it compares against user_id, not workspace_id, even
    # if the existing row happens to carry a workspace_id from a previous
    # workspace-scoped connect.
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    session = make_session(existing)
    with pytest.raises(CrossUserConflictError):
        await _upsert(USER_B, session)
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_merges_provider_metadata_instead_of_replacing_it():
    # The regression this guards. allowed_channel_ids is written only through
    # update_provider_metadata and appears in no token response, while
    # handle_slack_command reads an empty allowlist as "every channel
    # allowed". Replacing on reconnect therefore reopened, silently, a
    # workspace that had deliberately restricted the command to two channels.
    existing = make_existing(
        USER_A,
        STATUS_ACTIVE,
        provider_metadata={"bot_user_id": "U1", "allowed_channel_ids": ["C1", "C2"]},
    )
    session = make_session(existing)

    await _upsert(USER_A, session, provider_metadata={"bot_user_id": "U2"})

    # Keys the provider owns are refreshed; the one it cannot rebuild survives.
    assert existing.provider_metadata == {
        "bot_user_id": "U2",
        "allowed_channel_ids": ["C1", "C2"],
    }


@pytest.mark.asyncio
async def test_omitted_provider_metadata_leaves_the_stored_one_alone():
    # The member-link path upserts without metadata at all; that must not read
    # as a request to clear it.
    existing = make_existing(USER_A, STATUS_ACTIVE, provider_metadata={"bot_user_id": "U1"})
    session = make_session(existing)

    await _upsert(USER_A, session)

    assert existing.provider_metadata == {"bot_user_id": "U1"}


@pytest.mark.asyncio
async def test_first_connection_stores_the_metadata_it_was_given():
    # Nothing to merge into on an insert: the given dict is what lands.
    session = make_session(existing=None)

    credential = await _upsert(USER_A, session, provider_metadata={"account_login": "acme"})

    assert credential.provider_metadata == {"account_login": "acme"}


@pytest.mark.asyncio
async def test_omitted_workspace_id_leaves_the_stored_one_alone():
    # Ownership has to survive a reconnect by a caller that does not deal in
    # workspaces, or a shared connection quietly reverts to being user-owned.
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    session = make_session(existing)

    await _upsert(USER_A, session)

    assert existing.workspace_id == WORKSPACE_A


@pytest.mark.asyncio
async def test_explicit_none_workspace_id_still_clears_it():
    # The other half of the distinction: passing None is an instruction, and
    # it is still honored.
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    session = make_session(existing)

    await _upsert(USER_A, session, workspace_id=None)

    assert existing.workspace_id is None


@pytest.mark.asyncio
async def test_metadata_cannot_be_cleared_through_upsert():
    # Deliberately asymmetric with workspace_id above: merging leaves no way
    # to empty the metadata here, so neither None nor {} clears it. Anything
    # that needs to shrink the allowlist goes through update_provider_metadata.
    existing = make_existing(USER_A, STATUS_ACTIVE, provider_metadata={"allowed_channel_ids": []})
    session = make_session(existing)

    await _upsert(USER_A, session, provider_metadata=None)
    assert existing.provider_metadata == {"allowed_channel_ids": []}

    await _upsert(USER_A, session, provider_metadata={})
    assert existing.provider_metadata == {"allowed_channel_ids": []}


@pytest.mark.asyncio
async def test_get_active_credential_for_workspace_filters_by_workspace_and_status():
    existing = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    execute_result = MagicMock()
    execute_result.scalars.return_value.first.return_value = existing
    session = MagicMock()
    session.execute = AsyncMock(return_value=execute_result)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        result = await get_active_credential_for_workspace(WORKSPACE_A, PROVIDER)

    assert result is existing


@pytest.mark.asyncio
async def test_update_provider_metadata_merges_without_touching_token_fields():
    from cognee.modules.integrations.credentials import update_provider_metadata

    existing = make_existing(USER_A, STATUS_ACTIVE, provider_metadata={"bot_user_id": "U1"})
    session = make_session(existing)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        updated = await update_provider_metadata(
            PROVIDER, ACCOUNT_ID, {"allowed_channel_ids": ["C1"]}
        )

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


@pytest.mark.asyncio
async def test_record_sync_result_stamps_the_connection_it_ran_for():
    from cognee.modules.integrations.credentials import record_sync_result

    existing = make_existing(USER_A, STATUS_ACTIVE)
    existing.provider = PROVIDER
    existing.provider_account_id = ACCOUNT_ID
    session = make_session(existing)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(existing, status="ok")

    assert stamped is existing
    assert existing.sync_status == "ok"
    assert existing.last_synced_at is not None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_sync_that_outlived_its_install_does_not_stamp_the_new_owner():
    # Syncs run detached and upsert_credential reuses the row for a
    # (provider, account) rather than replacing it. So A disconnects, B
    # connects the same Google account, and A's sync finishes last: without
    # the owner check B would be told their Drive had just synced.
    from cognee.modules.integrations.credentials import record_sync_result

    ran_for = make_existing(USER_A, STATUS_ACTIVE)
    ran_for.provider = PROVIDER
    ran_for.provider_account_id = ACCOUNT_ID

    now_owned_by_b = make_existing(USER_B, STATUS_ACTIVE)
    now_owned_by_b.sync_status = None
    session = make_session(now_owned_by_b)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(ran_for, status="ok")

    assert stamped is None
    assert now_owned_by_b.sync_status is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_sync_finishing_after_a_disconnect_does_not_stamp_a_revoked_row():
    from cognee.modules.integrations.credentials import record_sync_result

    ran_for = make_existing(USER_A, STATUS_ACTIVE)
    ran_for.provider = PROVIDER
    ran_for.provider_account_id = ACCOUNT_ID

    disconnected = make_existing(USER_A, STATUS_REVOKED)
    disconnected.sync_status = None
    session = make_session(disconnected)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(ran_for, status="ok")

    assert stamped is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_token_refresh_mid_sync_does_not_block_the_stamp():
    # The owner check must not be so strict that the ordinary path trips it:
    # refreshing a token rewrites the row but keeps the same owner.
    #
    # This setup is indistinguishable from the same owner disconnecting and
    # reconnecting the same account, which is the gap record_sync_result's
    # own docstring names as still open — both are (USER_A, ACTIVE) to
    # (USER_A, ACTIVE) with no generation column to tell them apart. This
    # test only pins that the refresh case is accepted; it is not a claim
    # that the reconnect case is correctly rejected, because nothing here
    # can express that distinction yet.
    from cognee.modules.integrations.credentials import record_sync_result

    ran_for = make_existing(USER_A, STATUS_ACTIVE)
    ran_for.provider = PROVIDER
    ran_for.provider_account_id = ACCOUNT_ID

    after_refresh = make_existing(USER_A, STATUS_ACTIVE)
    session = make_session(after_refresh)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(ran_for, status="degraded")

    assert stamped is after_refresh
    assert after_refresh.sync_status == "degraded"


@pytest.mark.asyncio
async def test_a_sync_that_outlived_a_workspace_reconnect_does_not_stamp_the_new_workspace():
    # The owner check has to mirror upsert_credential's own owner
    # resolution, not just user_id: when workspace_id is set it is the
    # owner, the same way a workspace-scoped reconnect is judged. A check
    # that only compared user_id would miss the same misattribution on this
    # dimension — the same human reconnecting the same account under a
    # different workspace.
    from cognee.modules.integrations.credentials import record_sync_result

    ran_for = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_A)
    ran_for.provider = PROVIDER
    ran_for.provider_account_id = ACCOUNT_ID

    now_under_workspace_b = make_existing(USER_A, STATUS_ACTIVE, workspace_id=WORKSPACE_B)
    now_under_workspace_b.sync_status = None
    session = make_session(now_under_workspace_b)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(ran_for, status="ok")

    assert stamped is None
    assert now_under_workspace_b.sync_status is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_sync_still_stamps_a_row_that_switched_from_user_owned_to_workspace_owned():
    # The mirror image of the check above, to prove it is symmetric rather
    # than only ever refusing: nothing about the owner actually changed here
    # (no workspace_id before or after), so the stamp must still land.
    from cognee.modules.integrations.credentials import record_sync_result

    ran_for = make_existing(USER_A, STATUS_ACTIVE)
    ran_for.provider = PROVIDER
    ran_for.provider_account_id = ACCOUNT_ID

    unchanged = make_existing(USER_A, STATUS_ACTIVE)
    session = make_session(unchanged)

    with patch(
        "cognee.modules.integrations.credentials.get_relational_engine",
        return_value=make_engine(session),
    ):
        stamped = await record_sync_result(ran_for, status="ok")

    assert stamped is unchanged
    assert unchanged.sync_status == "ok"
