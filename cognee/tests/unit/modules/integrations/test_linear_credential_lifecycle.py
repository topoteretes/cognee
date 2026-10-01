"""Real SQL regressions for Linear's token refresh.

Linear access tokens last 24 hours and every refresh rotates the refresh token,
so these run against a fake token endpoint that rotates the same way: the
refresh token it just accepted stops working. A refresh that carried the old
token forward (as the Google adapters do) fails the second round here, which is
the silent disconnect about two days after connecting that the rotation
handling exists to prevent.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from sqlalchemy.exc import OperationalError

from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.linear import adapter
from cognee.modules.integrations.linear.adapter import LinearAuthError, LinearIntegration

ACCOUNT_ID = "org-1"


class FakeLinear:
    """Linear's token endpoint, as far as its docs describe it."""

    def __init__(self):
        self.valid = "refresh-0"
        self.generation = 0
        self.calls = []
        self.started = asyncio.Event()
        # Set to hold a refresh in flight while the test changes the row.
        self.gate = None
        # Linear replays a spent refresh token's request for 30 minutes. Off by
        # default so that carrying the old token forward still fails loudly.
        self.replay = False
        self.replayable = {}
        # An exception to raise after the next rotation, as a lost answer.
        self.lose_answer = None

    async def refresh(self, refresh_token, *, client_id, client_secret):
        self.calls.append(refresh_token)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.replay and refresh_token in self.replayable:
            return self.replayable[refresh_token]
        if refresh_token != self.valid:
            raise LinearAuthError("token refresh", "invalid_grant")
        self.generation += 1
        spent, self.valid = self.valid, f"refresh-{self.generation}"
        response = {
            "access_token": f"access-{self.generation}",
            "refresh_token": self.valid,
            "expires_in": 86399,
            "scope": "read write",
        }
        self.replayable[spent] = response
        if self.lose_answer is not None:
            failure, self.lose_answer = self.lose_answer, None
            raise failure
        return response


@pytest_asyncio.fixture
async def linear(credential_db, monkeypatch):
    fake = FakeLinear()
    monkeypatch.setattr(adapter, "require", lambda key: "test")
    monkeypatch.setattr(adapter, "refresh_access_token", fake.refresh)
    monkeypatch.setattr(adapter, "_INVALID_GRANT_SETTLE", 0)
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 0)
    monkeypatch.setattr(adapter, "_FAILURE_MEMORY", 0)
    # Retry state is module global: give each test its own and clean up after it.
    monkeypatch.setattr(adapter, "_retry_tasks", set())
    monkeypatch.setattr(adapter, "_pending_retries", set())
    monkeypatch.setattr(adapter, "_recent_failures", {})
    yield fake
    leftovers = list(adapter._retry_tasks)
    for task in leftovers:
        task.cancel()
    await asyncio.gather(*leftovers, return_exceptions=True)


async def install(user_id=None, *, token="access-0", refresh="refresh-0", expires_in=-1):
    """Connect the workspace. Expired by default, in hours from now."""
    token_payload = {"access_token": token}
    if refresh:
        token_payload["refresh_token"] = refresh
    return await store.upsert_credential(
        provider="linear",
        provider_account_id=ACCOUNT_ID,
        user_id=user_id or uuid4(),
        token_payload=token_payload,
        token_expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=expires_in)
            if expires_in is not None
            else None
        ),
    )


async def persisted():
    return await store.get_credential_by_account("linear", ACCOUNT_ID)


async def expire(credential):
    current = await store.require_active_credential(credential)
    await store.update_refreshed_credential(
        current,
        token_payload=store.decrypt_token_payload(current),
        token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        scopes=current.scopes,
    )


@pytest.mark.asyncio
async def test_an_expired_token_is_refreshed_on_use(linear):
    original = await install()

    assert await adapter.access_token_for(original) == "access-1"

    current = await persisted()
    assert store.decrypt_token_payload(current) == {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
    }
    assert current.scopes == "read write"
    # The new expiry is a day out, so the next caller does not refresh again.
    assert await adapter.access_token_for(original) == "access-1"
    assert linear.calls == ["refresh-0"]


@pytest.mark.asyncio
async def test_a_healthy_token_is_left_alone(linear):
    original = await install(expires_in=12)

    assert await adapter.access_token_for(original) == "access-0"
    assert linear.calls == []


@pytest.mark.asyncio
async def test_a_token_without_an_expiry_is_left_alone(linear):
    original = await install(expires_in=None)

    assert await adapter.access_token_for(original) == "access-0"
    assert linear.calls == []


@pytest.mark.asyncio
async def test_the_second_refresh_spends_the_refresh_token_the_first_returned(linear):
    original = await install()

    assert await adapter.access_token_for(original) == "access-1"
    await expire(original)
    assert await adapter.access_token_for(original) == "access-2"

    assert linear.calls == ["refresh-0", "refresh-1"]
    assert (await persisted()).status == "active"


@pytest.mark.asyncio
async def test_a_response_without_a_refresh_token_keeps_the_stored_one(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(return_value={"access_token": "access-1", "expires_in": 86399}),
    )

    await LinearIntegration().refresh(original)

    assert store.decrypt_token_payload(await persisted()) == {
        "access_token": "access-1",
        "refresh_token": "refresh-0",
    }


@pytest.mark.asyncio
async def test_concurrent_callers_trigger_one_refresh_and_none_fails(linear):
    original = await install()
    linear.gate = asyncio.Event()

    callers = [asyncio.create_task(adapter.access_token_for(original)) for _ in range(5)]
    await asyncio.wait_for(linear.started.wait(), 5)
    # Give the other callers time to reach the lock before the refresh finishes.
    await asyncio.sleep(0.05)
    linear.gate.set()

    assert await asyncio.gather(*callers) == ["access-1"] * 5
    assert linear.calls == ["refresh-0"]
    assert (await persisted()).status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("reconnect", [False, True])
async def test_refresh_cannot_resurrect_or_overwrite_reconnected_credentials(linear, reconnect):
    original = await install()
    linear.gate = asyncio.Event()
    pending = asyncio.create_task(LinearIntegration().refresh(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    await store.revoke_credential_by_account("linear", ACCOUNT_ID)
    if reconnect:
        await install(original.user_id, token="reconnected", refresh="refresh-new", expires_in=12)
    linear.gate.set()

    with pytest.raises(store.CredentialInactiveError):
        await pending

    current = await persisted()
    assert current.status == ("active" if reconnect else "revoked")
    assert store.decrypt_token_payload(current)["access_token"] == (
        "reconnected" if reconnect else "access-0"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reconnect", [False, True])
async def test_a_caller_racing_a_disconnect_fails_or_uses_the_reconnected_token(linear, reconnect):
    original = await install()
    linear.gate = asyncio.Event()
    pending = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    await store.revoke_credential_by_account("linear", ACCOUNT_ID)
    if reconnect:
        await install(original.user_id, token="reconnected", refresh="refresh-new", expires_in=12)
    linear.gate.set()

    if reconnect:
        assert await pending == "reconnected"
    else:
        with pytest.raises(store.CredentialInactiveError):
            await pending
        assert (await persisted()).status == "revoked"


@pytest.mark.asyncio
async def test_a_refresh_that_lost_the_race_uses_the_token_that_won(linear):
    """Another process refreshed first: the caller ends up with its token, not an error."""
    original = await install()
    linear.gate = asyncio.Event()
    pending = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    await store.update_refreshed_credential(
        original,
        token_payload={"access_token": "other-process", "refresh_token": "refresh-other"},
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        scopes=None,
    )
    linear.gate.set()

    assert await pending == "other-process"
    assert (await persisted()).status == "active"


@pytest.mark.asyncio
async def test_invalid_grant_revokes_the_credential(linear):
    original = await install(refresh="dead-refresh")

    with pytest.raises(LinearAuthError) as caught:
        await LinearIntegration().refresh(original)

    assert caught.value.code == "invalid_grant"
    assert (await persisted()).status == "revoked"


@pytest.mark.asyncio
async def test_invalid_grant_for_a_replaced_token_leaves_the_newer_credential_alone(linear):
    """Off contract: Linear rejects our token although another process stored a new one.

    Within its 30 minute replay window Linear should not do that, but if it ever
    does, revoking on the rejection would disconnect a healthy workspace.
    """
    original = await install()
    linear.valid = "spent-elsewhere"
    linear.gate = asyncio.Event()
    pending = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    await store.update_refreshed_credential(
        original,
        token_payload={"access_token": "other-process", "refresh_token": "refresh-other"},
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        scopes=None,
    )
    linear.gate.set()

    assert await pending == "other-process"
    assert (await persisted()).status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [LinearAuthError("token refresh", "http_500"), asyncio.TimeoutError()]
)
async def test_a_failure_that_is_not_invalid_grant_keeps_the_connection(
    linear, monkeypatch, failure
):
    original = await install()
    monkeypatch.setattr(adapter, "refresh_access_token", AsyncMock(side_effect=failure))

    with pytest.raises(type(failure)):
        await adapter.access_token_for(original)

    current = await persisted()
    assert current.status == "active"
    assert store.decrypt_token_payload(current)["refresh_token"] == "refresh-0"


@pytest.mark.asyncio
async def test_a_credential_without_a_refresh_token_says_to_reconnect(linear):
    original = await install(refresh=None)

    with pytest.raises(RuntimeError, match="must reconnect"):
        await adapter.access_token_for(original)

    assert linear.calls == []


@pytest.mark.asyncio
async def test_a_refresh_stores_the_expiry_linear_returned(linear):
    original = await install()

    await adapter.access_token_for(original)

    # FakeLinear answers expires_in=86399 seconds; SQLite hands the UTC value back naive.
    stored = (await persisted()).token_expires_at
    assert stored is not None
    remaining = stored.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) < remaining <= timedelta(seconds=86399)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expires_in", [None, 0, -100, "abc", True, 10**30, 1e12, float("inf"), float("nan")]
)
async def test_an_unusable_expires_in_is_assumed_to_last_a_day(linear, monkeypatch, expires_in):
    original = await install()
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(
            return_value={
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "expires_in": expires_in,
            }
        ),
    )

    await LinearIntegration().refresh(original)

    stored = (await persisted()).token_expires_at
    remaining = stored.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
    assert remaining > timedelta(hours=23, minutes=59)


@pytest.fixture
def host_zone(request):
    """Pin the local zone so a naive value read as local time fails on a UTC CI host too."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX only")
    previous = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    yield
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("minutes_left", "refreshes", "host_zone"),
    [(4, True, "America/Los_Angeles"), (6, False, "Asia/Tokyo")],
    indirect=["host_zone"],
)
async def test_the_refresh_margin_is_five_minutes(linear, host_zone, minutes_left, refreshes):
    original = await store.upsert_credential(
        provider="linear",
        provider_account_id=ACCOUNT_ID,
        user_id=uuid4(),
        token_payload={"access_token": "access-0", "refresh_token": "refresh-0"},
        token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=minutes_left),
    )

    token = await adapter.access_token_for(original)

    assert token == ("access-1" if refreshes else "access-0")
    assert linear.calls == (["refresh-0"] if refreshes else [])


@pytest.mark.asyncio
async def test_refresh_sends_the_configured_client_id_and_secret(linear, monkeypatch):
    original = await install()
    settings = {"client_id": "lin_client", "client_secret": "lin_secret"}
    monkeypatch.setattr(adapter, "require", settings.__getitem__)
    endpoint = AsyncMock(side_effect=linear.refresh)
    monkeypatch.setattr(adapter, "refresh_access_token", endpoint)

    await LinearIntegration().refresh(original)

    endpoint.assert_awaited_once_with(
        "refresh-0", client_id="lin_client", client_secret="lin_secret"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reconnect", [False, True])
async def test_a_healthy_token_follows_a_disconnect_or_reconnect(linear, reconnect):
    original = await install(expires_in=12)
    await store.revoke_credential_by_account("linear", ACCOUNT_ID)
    if reconnect:
        await install(original.user_id, token="reconnected", refresh="refresh-new", expires_in=12)

    if reconnect:
        assert await adapter.access_token_for(original) == "reconnected"
    else:
        with pytest.raises(store.CredentialInactiveError):
            await adapter.access_token_for(original)
    assert linear.calls == []


@pytest.mark.asyncio
async def test_a_payload_without_an_access_token_raises(linear):
    original = await install(token="", expires_in=12)

    with pytest.raises(RuntimeError, match="holds no access token"):
        await adapter.access_token_for(original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [LinearAuthError("token refresh", "http_500"), asyncio.TimeoutError(), aiohttp.ClientError()],
)
async def test_a_failed_refresh_falls_back_to_a_token_that_is_still_valid(
    linear, monkeypatch, failure
):
    original = await install(expires_in=3 / 60)  # three minutes left, inside the margin
    monkeypatch.setattr(adapter, "refresh_access_token", AsyncMock(side_effect=failure))

    assert await adapter.access_token_for(original) == "access-0"

    assert (await persisted()).status == "active"


@pytest.mark.asyncio
async def test_a_failed_refresh_of_a_dead_token_still_raises(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(side_effect=LinearAuthError("token refresh", "http_500")),
    )

    with pytest.raises(LinearAuthError):
        await adapter.access_token_for(original)


@pytest.mark.asyncio
async def test_callers_with_a_valid_token_do_not_queue_behind_a_slow_refresh(linear):
    original = await install(expires_in=3 / 60)
    linear.gate = asyncio.Event()
    refresher = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)

    # The others return the still valid token at once instead of waiting for the gate.
    others = await asyncio.wait_for(
        asyncio.gather(*(adapter.access_token_for(original) for _ in range(3))), 2
    )
    linear.gate.set()

    assert others == ["access-0"] * 3
    assert await refresher == "access-1"
    assert linear.calls == ["refresh-0"]


@pytest.mark.asyncio
async def test_invalid_grant_is_the_cause_of_the_inactive_error(linear):
    original = await install(refresh="dead-refresh")

    with pytest.raises(store.CredentialInactiveError) as caught:
        await adapter.access_token_for(original)

    assert isinstance(caught.value.__cause__, LinearAuthError)
    assert caught.value.__cause__.code == "invalid_grant"


@pytest.mark.asyncio
async def test_a_response_without_a_refresh_token_is_logged(linear, monkeypatch, caplog):
    original = await install()
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(return_value={"access_token": "access-1", "expires_in": 86399}),
    )

    with caplog.at_level("WARNING"):
        await LinearIntegration().refresh(original)

    assert "returned no refresh token" in caplog.text


async def drain_retries():
    while adapter._retry_tasks:
        await asyncio.gather(*list(adapter._retry_tasks))


@pytest.mark.asyncio
async def test_invalid_grant_waits_for_a_racing_process_before_it_revokes(linear, monkeypatch):
    """The other process commits during the settle delay, so the revoke finds a changed row."""
    original = await install()
    linear.valid = "spent-elsewhere"
    monkeypatch.setattr(adapter, "_INVALID_GRANT_SETTLE", 0.2)
    pending = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    await asyncio.sleep(0.05)  # the rejection has arrived and the settle delay is running
    await store.update_refreshed_credential(
        original,
        token_payload={"access_token": "other-process", "refresh_token": "refresh-other"},
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        scopes=None,
    )

    assert await pending == "other-process"
    assert (await persisted()).status == "active"


async def lose_the_first_answer(linear, failure, original):
    """Linear rotates, the caller never learns it, a retry replays the same request."""
    linear.replay = True
    linear.lose_answer = failure
    with pytest.raises(type(failure)):
        await adapter.access_token_for(original)
    await drain_retries()


def assert_recovered(linear, current):
    assert current.status == "active"
    assert store.decrypt_token_payload(current) == {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
    }
    assert linear.calls == ["refresh-0", "refresh-0"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        asyncio.TimeoutError(),
        aiohttp.ClientError(),
        LinearAuthError("token refresh", "http_200"),  # the body never finished
        LinearAuthError("token refresh", "http_504"),  # a gateway gave up after Linear rotated
    ],
)
async def test_a_lost_rotation_is_retried_with_the_same_refresh_token(linear, failure):
    original = await install()

    await lose_the_first_answer(linear, failure, original)

    assert_recovered(linear, await persisted())


@pytest.mark.asyncio
async def test_a_cancelled_refresh_is_retried(linear):
    original = await install()
    linear.replay = True
    linear.gate = asyncio.Event()
    pending = asyncio.create_task(adapter.access_token_for(original))
    await asyncio.wait_for(linear.started.wait(), 5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    linear.gate = None
    await drain_retries()

    assert linear.calls == ["refresh-0", "refresh-0"]
    assert store.decrypt_token_payload(await persisted())["access_token"] == "access-1"


@pytest.mark.asyncio
async def test_a_failed_save_is_retried_too(linear, monkeypatch):
    original = await install()
    linear.replay = True
    real_update = adapter.update_refreshed_credential
    saves = []

    async def flaky(*args, **kwargs):
        saves.append(1)
        if len(saves) == 1:
            raise OperationalError("database is locked", None, Exception())
        return await real_update(*args, **kwargs)

    monkeypatch.setattr(adapter, "update_refreshed_credential", flaky)

    with pytest.raises(OperationalError):
        await adapter.access_token_for(original)
    await drain_retries()

    assert_recovered(linear, await persisted())


@pytest.mark.asyncio
async def test_only_one_retry_is_pending_per_credential(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 3600)
    endpoint = AsyncMock(side_effect=asyncio.TimeoutError())
    monkeypatch.setattr(adapter, "refresh_access_token", endpoint)

    for _ in range(3):
        with pytest.raises(asyncio.TimeoutError):
            await adapter.access_token_for(original)

    assert len(adapter._retry_tasks) == 1
    assert endpoint.await_count == 3


@pytest.mark.asyncio
async def test_a_credential_gets_a_new_retry_after_the_last_one_finished(linear, monkeypatch):
    original = await install()
    endpoint = AsyncMock(side_effect=asyncio.TimeoutError())
    monkeypatch.setattr(adapter, "refresh_access_token", endpoint)

    for expected in (2, 4):  # each round: the caller, then its retry
        with pytest.raises(asyncio.TimeoutError):
            await adapter.access_token_for(original)
        await drain_retries()
        assert endpoint.await_count == expected


@pytest.mark.asyncio
async def test_a_retry_after_a_disconnect_calls_nothing(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 0.05)
    monkeypatch.setattr(
        adapter, "refresh_access_token", AsyncMock(side_effect=asyncio.TimeoutError())
    )
    with pytest.raises(asyncio.TimeoutError):
        await adapter.access_token_for(original)
    await store.revoke_credential_by_account("linear", ACCOUNT_ID)
    await drain_retries()

    assert adapter.refresh_access_token.await_count == 1
    assert (await persisted()).status == "revoked"


def test_the_delays_fit_the_windows_they_exist_for():
    # A retry has to land inside Linear's 30 minute replay window, and the wait
    # before an invalid_grant revoke plus the refresh timeout inside the 10
    # seconds Linear gives an agent session.
    assert 0 < adapter._RETRY_DELAY < 30 * 60
    # The failure memory must expire before the retry fires, or it would only answer from it.
    assert 0 < adapter._FAILURE_MEMORY < adapter._RETRY_DELAY
    assert adapter._REFRESH_TIMEOUT.total + adapter._INVALID_GRANT_SETTLE < 10


@pytest.mark.asyncio
async def test_a_token_that_just_died_is_not_handed_out_after_a_failed_refresh(linear, monkeypatch):
    original = await install(expires_in=-1 / 60)  # expired one minute ago
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(side_effect=LinearAuthError("token refresh", "http_500")),
    )

    with pytest.raises(LinearAuthError):
        await adapter.access_token_for(original)


@pytest.mark.asyncio
async def test_callers_fail_at_once_after_a_transient_failure_on_an_expired_token(
    linear, monkeypatch
):
    original = await install()
    monkeypatch.setattr(adapter, "_FAILURE_MEMORY", 60)
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 3600)
    endpoint = AsyncMock(side_effect=asyncio.TimeoutError())
    monkeypatch.setattr(adapter, "refresh_access_token", endpoint)

    for _ in range(4):
        with pytest.raises(asyncio.TimeoutError):
            await adapter.access_token_for(original)

    assert endpoint.await_count == 1


@pytest.mark.asyncio
async def test_a_successful_refresh_clears_the_failure_memory(linear):
    original = await install()
    # An old failure, outside the memory window, that the next success must wipe.
    adapter._recent_failures[original.id] = (
        adapter.time.monotonic() - 3600,
        asyncio.TimeoutError(),
    )

    assert await adapter.access_token_for(original) == "access-1"

    assert adapter._recent_failures == {}


@pytest.mark.asyncio
async def test_the_retry_ignores_a_failure_remembered_just_before_it_fires(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(adapter, "_FAILURE_MEMORY", 60)
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 0.2)
    linear.replay = True
    linear.lose_answer = asyncio.TimeoutError()
    with pytest.raises(asyncio.TimeoutError):
        await adapter.access_token_for(original)
    # Another caller failed moments before the retry fires.
    adapter._recent_failures[original.id] = (adapter.time.monotonic(), asyncio.TimeoutError())
    await drain_retries()

    assert_recovered(linear, await persisted())


@pytest.mark.asyncio
async def test_the_failure_memory_still_hands_out_a_token_that_is_valid(linear, monkeypatch):
    original = await install(expires_in=3 / 60)
    monkeypatch.setattr(adapter, "_FAILURE_MEMORY", 60)
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 3600)
    endpoint = AsyncMock(side_effect=asyncio.TimeoutError())
    monkeypatch.setattr(adapter, "refresh_access_token", endpoint)

    for _ in range(4):
        assert await adapter.access_token_for(original) == "access-0"

    assert endpoint.await_count == 1


@pytest.mark.asyncio
async def test_a_refresh_stores_an_expiry_that_differs_from_the_default(linear, monkeypatch):
    original = await install()
    monkeypatch.setattr(
        adapter,
        "refresh_access_token",
        AsyncMock(
            return_value={
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
            }
        ),
    )

    await LinearIntegration().refresh(original)

    stored = (await persisted()).token_expires_at
    remaining = stored.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
    assert timedelta(minutes=59) < remaining <= timedelta(hours=1)


@pytest.mark.asyncio
async def test_a_retry_after_a_disconnect_is_logged_quietly(linear, monkeypatch, caplog):
    original = await install()
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 0.05)
    monkeypatch.setattr(
        adapter, "refresh_access_token", AsyncMock(side_effect=asyncio.TimeoutError())
    )
    with pytest.raises(asyncio.TimeoutError):
        await adapter.access_token_for(original)
    await store.revoke_credential_by_account("linear", ACCOUNT_ID)
    with caplog.at_level("INFO"):
        await drain_retries()

    assert "connection is gone" in caplog.text
    assert not [record for record in caplog.records if record.levelname in ("ERROR", "CRITICAL")]


@pytest.mark.asyncio
async def test_a_normal_refresh_logs_no_missing_refresh_token_warning(linear, caplog):
    original = await install()

    with caplog.at_level("WARNING"):
        await adapter.access_token_for(original)

    assert "returned no refresh token" not in caplog.text


@pytest.mark.asyncio
async def test_a_retry_waiting_is_not_registered_for_the_shutdown_drain(linear, monkeypatch):
    from cognee.infrastructure import background_tasks

    original = await install()
    monkeypatch.setattr(adapter, "_RETRY_DELAY", 3600)
    monkeypatch.setattr(
        adapter, "refresh_access_token", AsyncMock(side_effect=asyncio.TimeoutError())
    )
    before = set(background_tasks._BACKGROUND_TASKS)

    with pytest.raises(asyncio.TimeoutError):
        await adapter.access_token_for(original)

    assert len(adapter._retry_tasks) == 1
    assert set(background_tasks._BACKGROUND_TASKS) == before


@pytest.mark.asyncio
async def test_a_retry_refreshing_is_registered_and_survives_being_cancelled(linear):
    from cognee.infrastructure import background_tasks

    original = await install()
    linear.replay = True
    linear.lose_answer = asyncio.TimeoutError()
    with pytest.raises(asyncio.TimeoutError):
        await adapter.access_token_for(original)
    linear.started.clear()
    linear.gate = asyncio.Event()  # holds the retry's request in flight
    before = set(background_tasks._BACKGROUND_TASKS)
    await asyncio.wait_for(linear.started.wait(), 5)

    assert len(set(background_tasks._BACKGROUND_TASKS) - before) == 1
    for task in list(adapter._retry_tasks):
        task.cancel()  # what a shutdown does to the retry task
    linear.gate.set()
    assert await background_tasks.wait_for_background_tasks(timeout=5)

    assert_recovered(linear, await persisted())
