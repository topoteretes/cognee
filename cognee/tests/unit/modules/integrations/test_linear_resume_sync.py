"""Cut-short Linear syncs are continued by a periodic worker."""

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

resume_module = importlib.import_module("cognee.modules.integrations.linear.resume_sync")

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def _credential(resume_needed_at, account="org-1"):
    return SimpleNamespace(
        provider_account_id=account,
        provider_metadata={"resume_needed_at": resume_needed_at},
    )


def _ago(minutes):
    return (NOW - timedelta(minutes=minutes)).isoformat()


@pytest.mark.parametrize(
    ("resume_needed_at", "expected"),
    [
        (_ago(30), True),
        (_ago(10), True),
        # the quota has not had time to refill
        (_ago(3), False),
        (None, False),
        ("", False),
        ("not a time", False),
        # a naive timestamp is read as UTC
        ((NOW - timedelta(minutes=30)).replace(tzinfo=None).isoformat(), True),
    ],
)
def test_only_a_pass_cut_short_by_the_quota_long_enough_ago_is_resumable(
    resume_needed_at, expected
):
    """The quota is a leaky bucket: a new attempt is pointless before some of it has refilled."""
    assert resume_module.is_resumable(_credential(resume_needed_at), NOW) is expected


def test_a_connection_without_the_marker_is_not_resumable():
    credential = SimpleNamespace(provider_account_id="org-1", provider_metadata=None)
    assert resume_module.is_resumable(credential, NOW) is False


@pytest.mark.asyncio
async def test_the_tick_resumes_only_active_linear_connections_that_are_due(
    credential_db, monkeypatch
):
    from cognee.modules.integrations import credentials as store

    owner = uuid4()

    async def connect(provider, account, marker, revoked=False):
        await store.upsert_credential(
            provider=provider,
            user_id=owner if provider == "linear" else uuid4(),
            provider_account_id=account,
            token_payload={"access_token": "t"},
            provider_metadata={"resume_needed_at": marker},
        )
        if revoked:
            await store.revoke_credential_by_account(provider, account)

    await connect("linear", "due", _ago(30))
    await connect("linear", "fresh", _ago(1))
    await connect("linear", "revoked", _ago(30), revoked=True)
    await connect("linear", "uncut", None)
    await connect("google_drive", "other-provider", _ago(30))

    monkeypatch.setattr(resume_module, "get_relational_engine", store.get_relational_engine)
    monkeypatch.setattr(resume_module, "datetime", _Frozen)
    request = AsyncMock(return_value=True)
    monkeypatch.setattr(resume_module, "request_sync", request)

    resumed = await resume_module.resume_cut_short_syncs()

    assert [call.args[0].provider_account_id for call in request.await_args_list] == ["due"]
    assert resumed == 1


@pytest.mark.asyncio
async def test_a_failing_or_dropped_resume_is_not_counted_and_stops_no_other(monkeypatch):
    """A sync that finds another one running returns False and is not counted as resumed."""
    due = [_credential(_ago(30), account) for account in ("a", "b", "c")]

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, statement):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: due))

    monkeypatch.setattr(
        resume_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _Session()),
    )
    monkeypatch.setattr(resume_module, "datetime", _Frozen)

    async def request(credential):
        if credential.provider_account_id == "a":
            raise RuntimeError("pod down")
        return credential.provider_account_id == "b"  # c finds a sync already running

    monkeypatch.setattr(resume_module, "request_sync", request)

    assert await resume_module.resume_cut_short_syncs() == 1


@pytest.mark.asyncio
async def test_the_syncs_of_different_connections_run_side_by_side(monkeypatch):
    """One workspace's long first sync must not hold up the other workspaces' resumes."""
    due = [_credential(_ago(30), account) for account in ("a", "b")]

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, statement):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: due))

    monkeypatch.setattr(
        resume_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _Session()),
    )
    monkeypatch.setattr(resume_module, "datetime", _Frozen)
    running = set()
    overlapped = []

    async def request(credential):
        running.add(credential.provider_account_id)
        await asyncio.sleep(0.01)
        overlapped.append(len(running))
        running.discard(credential.provider_account_id)
        return True

    monkeypatch.setattr(resume_module, "request_sync", request)

    await resume_module.resume_cut_short_syncs()

    assert max(overlapped) == 2


def test_the_integrations_router_starts_the_worker_only_when_linear_is_configured(monkeypatch):
    from cognee.api.v1.integrations.routers.get_integrations_router import (
        get_integrations_router,
    )

    started = []

    async def fake_worker():
        started.append(True)

    monkeypatch.setattr(resume_module, "_worker", fake_worker)
    monkeypatch.setattr(resume_module.linear_settings, "resume_sync_enabled", True)

    for client_id, expected in (("", 0), ("client", 1)):
        started.clear()
        monkeypatch.setattr(resume_module.linear_settings, "client_id", client_id)
        app = FastAPI()
        app.include_router(get_integrations_router(), prefix="/integrations")
        with TestClient(app):
            pass
        assert len(started) == expected
