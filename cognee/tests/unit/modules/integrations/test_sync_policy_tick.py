"""The tick starts each due slot once and only the connections it covers (SDK-1434).

Runs on a real SQLite file (the ``database`` fixture in ``conftest.py``) with a fake
host: ``start`` records the connection and no provider is involved.
"""

import asyncio
import importlib
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.modules.integrations.sync_policy import SyncPolicy
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy

store = importlib.import_module("cognee.modules.integrations.sync_policy.store")
tick = importlib.import_module("cognee.modules.integrations.sync_policy.tick")

SAVED = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
# Two days of hourly slots later: every policy saved at SAVED is due.
NOW = SAVED + timedelta(days=2)
HOURLY = SyncPolicy(interval_minutes=60)
RACERS = 6


class _Host:
    """A fake host: the owner's connections, and a start that only records."""

    def __init__(self, owned=()):
        self.owned = list(owned)
        self.started = []

    async def owned_credential_ids(self, principal_id):
        return self.owned

    async def start(self, credential_id):
        self.started.append(credential_id)

    async def tick(self, now=NOW):
        return await tick.run_due_syncs(now, self.owned_credential_ids, self.start)


def _owner():
    return SimpleNamespace(id=uuid4())


async def _row(database, row_id):
    async with database.get_async_session() as session:
        return await session.get(IntegrationSyncPolicy, row_id)


async def _next_due(database, row_id):
    # SQLite reads the stored UTC slot back without its offset.
    return (await _row(database, row_id)).next_due_at.replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_due_default_starts_the_connections_without_their_own_row(database):
    owner = _owner()
    inheriting, overriding = uuid4(), uuid4()
    await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    # Its own row is not due yet, and it never runs on the default's slot.
    await store.save_sync_policy(owner, owner.id, HOURLY, overriding, now=NOW)
    host = _Host(owned=[inheriting, overriding])

    assert await host.tick() == 1
    assert host.started == [inheriting]


@pytest.mark.asyncio
async def test_missed_slots_are_one_run_and_the_next_slot_counts_from_now(database):
    owner = _owner()
    row = await store.save_sync_policy(owner, owner.id, HOURLY, uuid4(), now=SAVED)
    host = _Host(owned=[row.credential_id])

    await host.tick()
    await host.tick()

    assert host.started == [row.credential_id]
    assert await _next_due(database, row.id) == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_paused_row_is_not_claimed_and_its_slot_waits(database):
    owner = _owner()
    row = await store.save_sync_policy(owner, owner.id, HOURLY, uuid4(), now=SAVED)
    await store.pause_sync_policy(owner, owner.id, row.credential_id, now=SAVED)
    host = _Host(owned=[row.credential_id])

    assert await host.tick() == 0
    assert await _next_due(database, row.id) == datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)

    await store.resume_sync_policy(owner, owner.id, row.credential_id, now=NOW)
    assert await host.tick() == 0
    assert await host.tick(NOW + timedelta(hours=1)) == 1


@pytest.mark.asyncio
async def test_a_paused_default_holds_a_connection_with_its_own_row_and_its_slots_pass(database):
    owner = _owner()
    await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    own = await store.save_sync_policy(owner, owner.id, HOURLY, uuid4(), now=SAVED)
    await store.pause_sync_policy(owner, owner.id, now=SAVED)
    host = _Host(owned=[own.credential_id])

    assert await host.tick() == 0
    # Claimed without a run, so resuming the default does not start a burst of old slots.
    assert await _next_due(database, own.id) == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_an_own_row_starts_nothing_once_its_principal_no_longer_owns_the_connection(
    database,
):
    """A reconnect can hand the credential row to another owner; the old row must not
    schedule the new owner's connection. Its slot still passes."""
    former_owner = _owner()
    row = await store.save_sync_policy(former_owner, former_owner.id, HOURLY, uuid4(), now=SAVED)
    host = _Host(owned=[])

    assert await host.tick() == 0
    assert await _next_due(database, row.id) == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_default_paused_between_the_read_and_the_claim_still_holds_the_connection(
    database, monkeypatch
):
    owner = _owner()
    await store.save_sync_policy(owner, owner.id, HOURLY, now=NOW)
    own = await store.save_sync_policy(owner, owner.id, HOURLY, uuid4(), now=SAVED)
    host = _Host(owned=[own.credential_id])
    claim = tick.claim_due_slot

    async def pause_then_claim(row, now):
        await store.pause_sync_policy(owner, owner.id, now=NOW)
        return await claim(row, now)

    monkeypatch.setattr(tick, "claim_due_slot", pause_then_claim)

    assert await host.tick() == 0
    assert await _next_due(database, own.id) == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_slot_is_claimed_once_against_the_value_sqlite_reads_back(database):
    """SQLite returns the stored UTC slot without its offset; the claim must still match it."""
    owner = _owner()
    saved = await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    read_back = await _row(database, saved.id)
    assert read_back.next_due_at.tzinfo is None

    assert await tick.claim_due_slot(read_back, NOW) is True
    assert await tick.claim_due_slot(read_back, NOW) is False


@pytest.mark.asyncio
async def test_a_pause_between_the_due_scan_and_the_claim_wins(database):
    owner = _owner()
    saved = await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    scanned = await _row(database, saved.id)
    await store.pause_sync_policy(owner, owner.id, now=SAVED)

    assert await tick.claim_due_slot(scanned, NOW) is False


@pytest.mark.asyncio
async def test_a_failed_read_of_the_connections_leaves_the_slot_due(database):
    owner = _owner()
    inheriting = uuid4()
    await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    host = _Host(owned=[inheriting])
    listed = host.owned_credential_ids

    async def failing_once(principal_id):
        host.owned_credential_ids = listed
        raise ConnectionRefusedError("database restarting")

    host.owned_credential_ids = failing_once
    with pytest.raises(ConnectionRefusedError):
        await host.tick()

    assert await host.tick() == 1
    assert host.started == [inheriting]


@pytest.mark.asyncio
async def test_an_invalid_row_is_skipped_and_the_others_still_run(database):
    owner = _owner()
    valid = await store.save_sync_policy(owner, owner.id, HOURLY, uuid4(), now=SAVED)
    database.execute(
        "INSERT INTO integration_sync_policies (id, principal_id, credential_id,"
        " interval_minutes, weekdays, timezone, paused, next_due_at, created_at, updated_at)"
        " VALUES (?, ?, ?, 0, '[0]', 'UTC', 0, '2026-11-05 00:00:00.000000',"
        " '2026-11-05 00:00:00', '2026-11-05 00:00:00')",
        (uuid4().hex, owner.id.hex, uuid4().hex),
    )
    host = _Host(owned=[valid.credential_id])

    assert await host.tick() == 1
    assert host.started == [valid.credential_id]


_RACER = textwrap.dedent(
    """
    import asyncio, sys, time
    from datetime import datetime, timezone
    from pathlib import Path
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from types import SimpleNamespace
    from cognee.modules.integrations.sync_policy import tick
    from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy

    db_path, ready, go = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

    async def main():
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        tick.get_relational_engine = lambda: SimpleNamespace(get_async_session=sessions)
        async with sessions() as session:
            row = (await session.execute(select(IntegrationSyncPolicy))).scalar_one()
        # Every racer holds the same slot before any of them claims it.
        ready.touch()
        while not go.exists():
            await asyncio.sleep(0.01)
        now = datetime.fromisoformat(sys.argv[4])
        print(int(await tick.claim_due_slot(row, now)))
        await engine.dispose()

    asyncio.run(main())
    """
)


@pytest.mark.asyncio
async def test_racing_processes_claim_one_slot_exactly_once(database, tmp_path):
    owner = _owner()
    await store.save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    go = tmp_path / "go"
    ready = [tmp_path / f"ready-{index}" for index in range(RACERS)]
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(database.path), str(flag), str(go), NOW.isoformat()],
            stdout=subprocess.PIPE,
            text=True,
        )
        for flag in ready
    ]
    try:
        # Each racer exits on an error, so waiting for its flag cannot outlive it.
        while not all(flag.exists() for flag in ready):
            if any(racer.poll() is not None for racer in racers):
                break
            await asyncio.sleep(0.05)
        go.touch()
        outputs = [racer.communicate(timeout=120)[0].strip() for racer in racers]
    finally:
        for racer in racers:
            racer.kill()

    assert sorted(outputs) == ["0"] * (RACERS - 1) + ["1"]
    async with database.get_async_session() as session:
        row = (await session.execute(select(IntegrationSyncPolicy))).scalar_one()
    assert await _next_due(database, row.id) == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
