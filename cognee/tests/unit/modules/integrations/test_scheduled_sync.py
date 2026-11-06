"""The SDK API server's side of scheduled sync: owners, starts and the worker (SDK-1434).

Real credential rows on one SQLite file (the ``database`` fixture in ``conftest.py``);
only the provider is fake.
"""

import asyncio
import base64
import importlib
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.background_tasks import wait_for_background_tasks
from cognee.modules.integrations.base import OAuthIntegration
from cognee.modules.integrations.credentials import (
    revoke_credential_by_account,
    upsert_credential,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.integrations.sync_policy import SyncPolicy, save_sync_policy
from cognee.modules.operations import ORIGIN_BACKGROUND, get_operation_origin

scheduled_sync = importlib.import_module("cognee.modules.integrations.scheduled_sync")

# A short tick: zero would let the worker loop without ever yielding to the test.
TICK = 0.01


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def __init__(self, sync_now):
        self.sync_now = sync_now

    def authorize_url(self, state):
        return ""

    async def exchange_code(self, code):
        return {}

    def parse_installation(self, token_response):
        raise NotImplementedError

    def state_signing_secret(self):
        return "secret"

    def frontend_base_url(self):
        return ""


@pytest.fixture
def provider():
    before = dict(supported_integrations)
    sync_now = AsyncMock()
    use_integration(_FakeIntegration(sync_now))
    yield sync_now
    supported_integrations.clear()
    supported_integrations.update(before)


async def _credential(database, user_id, workspace_id=None, status="active", provider="fake"):
    credential = IntegrationCredential(
        user_id=user_id,
        workspace_id=workspace_id,
        provider=provider,
        provider_account_id=str(uuid4()),
        ciphertext=b"x",
        nonce=b"x",
        status=status,
    )
    async with database.get_async_session() as session:
        session.add(credential)
        await session.commit()
    return credential.id


@pytest.mark.asyncio
async def test_a_workspace_owns_its_connections_and_a_user_only_those_without_one(database):
    user, workspace = uuid4(), uuid4()
    personal = await _credential(database, user)
    shared = await _credential(database, user, workspace_id=workspace)
    await _credential(database, user, status="revoked")

    assert await scheduled_sync.owned_credential_ids(user) == [personal]
    assert await scheduled_sync.owned_credential_ids(workspace) == [shared]
    assert scheduled_sync.owner_principal_id(
        SimpleNamespace(user_id=user, workspace_id=workspace)
    ) == (workspace)


@pytest.mark.asyncio
async def test_start_runs_the_providers_sync_now_in_the_background(database, provider):
    credential_id = await _credential(database, uuid4())

    await scheduled_sync.start_sync(credential_id)
    await wait_for_background_tasks()

    provider.assert_awaited_once()
    assert provider.await_args.args[0].id == credential_id


@pytest.mark.asyncio
async def test_a_failed_sync_is_logged_with_its_connection(database, provider, caplog):
    credential_id = await _credential(database, uuid4())
    provider.side_effect = RuntimeError("provider down")

    with caplog.at_level(logging.ERROR):
        await scheduled_sync.start_sync(credential_id)
        await wait_for_background_tasks()

    assert str(credential_id) in caplog.text


@pytest.mark.asyncio
async def test_a_disconnected_or_unregistered_connection_is_not_started(database, provider, caplog):
    revoked = await _credential(database, uuid4(), status="revoked")
    unregistered = await _credential(database, uuid4(), provider="gone")

    with caplog.at_level(logging.WARNING):
        await scheduled_sync.start_sync(revoked)
        await scheduled_sync.start_sync(unregistered)
        await wait_for_background_tasks()

    provider.assert_not_awaited()
    # Skipped, not failed: one warning for the provider nobody registered, no error.
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [logging.WARNING]
    assert "gone" in caplog.text and str(unregistered) in caplog.text


@pytest.mark.asyncio
async def test_the_tick_syncs_a_users_connections_from_the_default_end_to_end(database, provider):
    user = SimpleNamespace(id=uuid4())
    credential_id = await _credential(database, user.id)
    saved = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
    await save_sync_policy(user, user.id, SyncPolicy(interval_minutes=60), now=saved)

    started = await scheduled_sync.run_due_syncs(
        saved + timedelta(hours=1),
        scheduled_sync.owned_credential_ids,
        scheduled_sync.start_sync,
    )
    await wait_for_background_tasks()

    assert started == 1
    assert provider.await_args.args[0].id == credential_id


@pytest.mark.asyncio
async def test_the_worker_is_off_unless_enabled(monkeypatch):
    tick = AsyncMock()
    monkeypatch.setattr(scheduled_sync, "_tick_until", tick)
    monkeypatch.delenv("INTEGRATIONS_SYNC_SCHEDULER_ENABLED", raising=False)

    async with scheduled_sync.sync_scheduler_lifespan(None):
        await asyncio.sleep(0)
    tick.assert_not_awaited()

    monkeypatch.setenv("INTEGRATIONS_SYNC_SCHEDULER_ENABLED", "true")
    async with scheduled_sync.sync_scheduler_lifespan(None):
        await asyncio.sleep(0)
    tick.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_worker_outlives_a_failed_tick(monkeypatch):
    """A database restart raises driver errors SQLAlchemy does not wrap; the next tick runs."""
    origins = []
    second = asyncio.Event()

    async def run_due_syncs(now, owned, start):
        origins.append(get_operation_origin())
        if len(origins) == 1:
            raise ConnectionRefusedError("database restarting")
        second.set()
        return 0

    monkeypatch.setattr(scheduled_sync, "run_due_syncs", run_due_syncs)
    monkeypatch.setattr(scheduled_sync, "TICK_SECONDS", TICK)
    monkeypatch.setenv("INTEGRATIONS_SYNC_SCHEDULER_ENABLED", "true")

    async with scheduled_sync.sync_scheduler_lifespan(None):
        await asyncio.wait_for(second.wait(), timeout=5)

    # Scheduled runs are recorded as background work, not as an SDK call.
    assert origins[:2] == [ORIGIN_BACKGROUND, ORIGIN_BACKGROUND]


@pytest.mark.asyncio
async def test_shutdown_lets_a_tick_in_flight_finish_through_the_apps_drain(monkeypatch, caplog):
    """A tick cut off between its claim and its starts would lose the slot, and one that
    hangs on the database must not hold shutdown past the app's bounded drain."""
    entered, release, finished = asyncio.Event(), asyncio.Event(), []

    async def run_due_syncs(now, owned, start):
        entered.set()
        await release.wait()
        finished.append(True)
        return 0

    monkeypatch.setattr(scheduled_sync, "run_due_syncs", run_due_syncs)
    monkeypatch.setattr(scheduled_sync, "TICK_SECONDS", TICK)
    monkeypatch.setenv("INTEGRATIONS_SYNC_SCHEDULER_ENABLED", "true")

    with caplog.at_level(logging.INFO):
        async with scheduled_sync.sync_scheduler_lifespan(None):
            await asyncio.wait_for(entered.wait(), timeout=5)
        # The lifespan returned at once; the tick is still in flight, cut off by nothing.
        assert finished == []
        # The app's shutdown drain waits for it, within its own bound.
        assert not await wait_for_background_tasks(timeout=TICK)
        release.set()
        assert await wait_for_background_tasks(timeout=5)

    assert finished == [True]
    assert "Scheduled sync worker stopped" in caplog.text


@pytest.mark.asyncio
async def test_a_reconnected_account_is_synced_by_its_new_owners_policy_only(
    database, provider, monkeypatch
):
    """Reconnecting reuses the credential row; the previous owner's own policy for it must
    not schedule the new owner's connection, and the new owner's default must."""
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    former, current = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    connected = await upsert_credential(
        provider="fake", user_id=former.id, provider_account_id="ACCOUNT", token_payload={}
    )
    saved = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
    await save_sync_policy(former, former.id, SyncPolicy(interval_minutes=60), connected.id, saved)
    await revoke_credential_by_account("fake", "ACCOUNT")
    reconnected = await upsert_credential(
        provider="fake", user_id=current.id, provider_account_id="ACCOUNT", token_payload={}
    )
    assert reconnected.id == connected.id

    async def tick(now):
        started = await scheduled_sync.run_due_syncs(
            now, scheduled_sync.owned_credential_ids, scheduled_sync.start_sync
        )
        await wait_for_background_tasks()
        return started

    assert await tick(saved + timedelta(hours=1)) == 0
    provider.assert_not_awaited()

    await save_sync_policy(current, current.id, SyncPolicy(interval_minutes=60), now=saved)
    assert await tick(saved + timedelta(hours=2)) == 1
    assert provider.await_args.args[0].user_id == current.id
