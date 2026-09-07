from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack import adapter, history


@pytest.mark.asyncio
async def test_expiring_history_token_is_refreshed_before_fetch(monkeypatch):
    credential = SimpleNamespace(
        id=uuid4(),
        provider_account_id="T1",
        user_id=uuid4(),
        status="active",
        token_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    monkeypatch.setattr(history, "get_by_team", AsyncMock(return_value=credential))
    refresh = AsyncMock()
    monkeypatch.setattr(history.SlackIntegration, "refresh", refresh)
    monkeypatch.setattr(
        history, "decrypt_token_payload", lambda credential: {"access_token": "fresh"}
    )
    assert await history._access_token(credential) == "fresh"
    refresh.assert_awaited_once_with(credential)


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", [False, True])
async def test_token_rotation_preserves_sync_settings_and_cannot_revive_revocation(
    monkeypatch, revoked
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user_id, workspace_id = uuid4(), uuid4()
    metadata = {
        "installed_by_slack_user_id": "U1",
        "allowed_channel_ids": ["C1"],
        "history_sync": {"dataset": {"enabled": True}},
    }
    async with sessions() as db:
        credential = IntegrationCredential(
            user_id=user_id,
            workspace_id=workspace_id,
            provider="slack",
            provider_account_id="T1",
            scopes="channels:history",
            provider_metadata=metadata,
            ciphertext=b"old",
            nonce=b"nonce",
            status="active",
        )
        db.add(credential)
        await db.commit()
        if revoked:
            # Keep the detached input active, as it was before a slow HTTP call.
            from sqlalchemy import update

            await db.execute(
                update(IntegrationCredential)
                .where(
                    IntegrationCredential.id == credential.id,
                )
                .values(status="revoked"),
                execution_options={"synchronize_session": False},
            )
            await db.commit()

    monkeypatch.setattr(
        adapter, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    monkeypatch.setattr(
        adapter, "decrypt_token_payload", lambda credential: {"refresh_token": "old-refresh"}
    )
    monkeypatch.setattr(
        adapter, "encrypt_credentials", lambda payload: (b"new", b"new-nonce", 1, "1")
    )
    monkeypatch.setattr(adapter, "require", lambda name: "test-only")
    response = AsyncMock()
    response.__aenter__.return_value = response
    # Refresh intentionally lacks team, user and enterprise metadata.
    response.json.return_value = {
        "ok": True,
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 43200,
    }
    session = MagicMock()
    session.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(adapter.aiohttp, "ClientSession", lambda **kwargs: context)
    try:
        if revoked:
            with pytest.raises(RuntimeError, match="changed"):
                await adapter.SlackIntegration().refresh(credential)
        else:
            await adapter.SlackIntegration().refresh(credential)
        async with sessions() as db:
            stored = await db.get(IntegrationCredential, credential.id)
            assert stored.provider_metadata == metadata
            assert stored.workspace_id == workspace_id and stored.user_id == user_id
            assert stored.provider_account_id == "T1"
            assert stored.ciphertext == (b"old" if revoked else b"new")
            assert stored.status == ("revoked" if revoked else "active")
    finally:
        await engine.dispose()
