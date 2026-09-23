"""Real SQL regressions for refresh/disconnect races in both Google providers."""

import asyncio
import base64
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


@pytest_asyncio.fixture
async def credential_db(monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        store, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    yield
    await engine.dispose()


async def install(provider, user_id=None, token="original"):
    return await store.upsert_credential(
        provider=provider,
        provider_account_id="google-subject",
        user_id=user_id or uuid4(),
        token_payload={"access_token": token, "refresh_token": "refresh"},
        provider_metadata={"selected_folder_ids": ["one"]},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google_drive", "gmail"])
@pytest.mark.parametrize("reconnect", [False, True])
async def test_refresh_cannot_resurrect_or_overwrite_reconnected_credentials(
    credential_db, monkeypatch, provider, reconnect
):
    adapter = import_module(f"cognee.modules.integrations.{provider}.adapter")
    integration = getattr(
        adapter,
        "GoogleDriveIntegration" if provider == "google_drive" else "GoogleGmailIntegration",
    )()
    original = await install(provider)
    started, finish = asyncio.Event(), asyncio.Event()

    async def refresh(*args, **kwargs):
        started.set()
        await finish.wait()
        return {"access_token": "stale-refresh-result", "expires_in": 3600}

    monkeypatch.setattr(adapter, "require", lambda key: "test")
    monkeypatch.setattr(adapter.client, "refresh_access_token", refresh)
    pending = asyncio.create_task(integration.refresh(original))
    await started.wait()
    await store.revoke_credential_by_account(provider, original.provider_account_id)
    if reconnect:
        await install(provider, original.user_id, token="reconnected")
    finish.set()
    with pytest.raises(store.CredentialInactiveError):
        await pending
    persisted = await store.get_credential_by_account(provider, original.provider_account_id)
    assert persisted.status == ("active" if reconnect else "revoked")
    assert store.decrypt_token_payload(persisted)["access_token"] == (
        "reconnected" if reconnect else "original"
    )
    if not reconnect:
        with pytest.raises(store.CredentialInactiveError):
            await adapter.access_token_for(original)


@pytest.mark.asyncio
async def test_old_invalid_grant_cannot_revoke_new_installation(credential_db):
    original = await install("gmail")
    await install("gmail", original.user_id, token="reconnected")
    await store.revoke_credential_if_current(original)
    assert (await store.require_active_credential(original)).status == "active"


@pytest.mark.asyncio
async def test_successful_refresh_preserves_selection_and_health(credential_db):
    original = await install("gmail")
    await store.update_provider_metadata(
        "gmail", original.provider_account_id, {"selection": ["new"]}
    )
    await store.update_refreshed_credential(
        original, token_payload={"access_token": "fresh"}, token_expires_at=None, scopes="read"
    )
    current = await store.require_active_credential(original)
    assert store.decrypt_token_payload(current) == {"access_token": "fresh"}
    assert current.provider_metadata["selection"] == ["new"]


@pytest.mark.asyncio
async def test_extraction_checkpoint_stops_next_iteration_after_disconnect(credential_db):
    from cognee.modules.integrations.google.ingestion import extraction_checkpoint
    from cognee.tasks.ingestion.dlt_utils import guarded_rows

    original = await install("gmail")
    extracted = []

    def rows():
        for i in range(3):
            extracted.append(i)
            yield {"id": i}

    iterator = guarded_rows(rows(), extraction_checkpoint(original))
    assert await asyncio.to_thread(next, iterator) == {"id": 0}
    await store.revoke_credential_by_account("gmail", original.provider_account_id)
    with pytest.raises(store.CredentialInactiveError):
        await asyncio.to_thread(next, iterator)
    assert extracted == [0]


@pytest.mark.asyncio
async def test_sync_never_starts_with_detached_revoked_credential(credential_db):
    from cognee.modules.integrations.google.ingestion import run_sync

    original = await install("gmail")
    await store.revoke_credential_by_account("gmail", original.provider_account_id)
    source = AsyncMock()
    with pytest.raises(store.CredentialInactiveError):
        await run_sync("gmail", original, source)
    source.assert_not_awaited()
