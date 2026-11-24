"""Sync policies on the configured relational database, with no mocks but the provider.

SDK-1434 asks for the cross-process claim and the end-to-end tick to run on SQLite and
on Postgres in CI. The unit tests cover SQLite only; this suite runs on whatever the
environment configures (``DB_PROVIDER``, ``DB_HOST``, ...), so the same file guards
both: the 'Sync policy e2e' jobs in e2e_tests.yml run it once per backend.

A fresh database gets its schema from cognee's own migration chain, as in CI:
``create_db_and_tables`` runs the chain only on an empty database, so a local database
built before the sync-policy migration needs ``run_migrations()`` first. Users come from
``create_user`` and connections from ``upsert_credential``. The provider's sync is replaced by a recorder,
because what is under test is which connections a tick starts, not the provider.
"""

import asyncio
import base64
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

# Registers the credential model the policy table's foreign key names.
import cognee.modules.integrations.models
from cognee.infrastructure.databases.relational import create_db_and_tables
from cognee.modules.integrations.credentials import (
    list_active_credential_ids_for_owner,
    upsert_credential,
)
from cognee.modules.integrations.sync_policy import (
    SyncPolicy,
    pause_sync_policy,
    run_due_syncs,
    save_sync_policy,
)
from cognee.modules.users.methods import create_user

RACERS = 8
READY_SECONDS = 120
SAVED = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
HOURLY = SyncPolicy(interval_minutes=60)

_RACER = textwrap.dedent(
    """
    import asyncio, sys
    from datetime import datetime
    from pathlib import Path
    from uuid import UUID

    # Registers the credential model the policy table's foreign key names.
    import cognee.modules.integrations.models
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.integrations.sync_policy import tick
    from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy

    row_id, ready, go, now = UUID(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]

    async def main():
        async with get_relational_engine().get_async_session() as session:
            row = await session.get(IntegrationSyncPolicy, row_id)
        # Every racer holds the same slot before any of them claims it.
        ready.touch()
        while not go.exists():
            await asyncio.sleep(0.01)
        claimed = await tick.claim_due_slot(row, datetime.fromisoformat(now))
        print(f"CLAIMED={int(claimed)}")

    asyncio.run(main())
    """
)


async def _user():
    return await create_user(f"sync-policy-{uuid4().hex}@example.com", None)


async def _connection(user, account: str):
    credential = await upsert_credential(
        provider="fake", user_id=user.id, provider_account_id=account, token_payload={}
    )
    return credential.id


def _claimed(output: str) -> str:
    # The racer's own line; cognee may log to stdout before it.
    return next(line for line in output.splitlines() if line.startswith("CLAIMED="))


async def _race(row_id, now: datetime, workdir: Path) -> list[str]:
    go = workdir / "go"
    ready = [workdir / f"ready-{index}" for index in range(RACERS)]
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(row_id), str(flag), str(go), now.isoformat()],
            stdout=subprocess.PIPE,
            text=True,
        )
        for flag in ready
    ]
    try:
        # A racer that fails exits; one that hangs is caught by the deadline.
        deadline = time.monotonic() + READY_SECONDS
        while not all(flag.exists() for flag in ready):
            if any(racer.poll() is not None for racer in racers):
                break
            assert time.monotonic() < deadline, "racers did not get ready"
            await asyncio.sleep(0.05)
        go.touch()
        outputs = [racer.communicate(timeout=120)[0] for racer in racers]
    finally:
        for racer in racers:
            racer.kill()
    assert all(racer.returncode == 0 for racer in racers), outputs
    return sorted(_claimed(output) for output in outputs)


@pytest.fixture(scope="module")
def event_loop():
    """One loop for the module: cognee's relational engine is cached per process, and on
    Postgres its connections belong to the loop that opened them."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def database(monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    await create_db_and_tables()


@pytest.mark.asyncio
async def test_processes_holding_one_slot_claim_it_exactly_once(database, tmp_path):
    owner = await _user()
    raced = await save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    # Each claim moves the row to the next hour, so every round races a fresh slot.
    for round_index in range(3):
        due = raced.next_due_at + timedelta(hours=round_index)
        workdir = tmp_path / f"round-{round_index}"
        workdir.mkdir()
        assert await _race(raced.id, due, workdir) == ["CLAIMED=0"] * (RACERS - 1) + ["CLAIMED=1"]


@pytest.mark.asyncio
async def test_the_tick_starts_only_the_connections_a_due_unpaused_policy_covers(database):
    owner, stranger, held_owner = await _user(), await _user(), await _user()
    inheriting = await _connection(owner, f"inheriting-{uuid4().hex}")
    held = await _connection(owner, f"held-{uuid4().hex}")
    others = await _connection(stranger, f"others-{uuid4().hex}")
    held_by_default = await _connection(held_owner, f"held-by-default-{uuid4().hex}")

    # An owner's default starts the connections that inherit it, not one held by its
    # own paused policy.
    await save_sync_policy(owner, owner.id, HOURLY, now=SAVED)
    await save_sync_policy(owner, owner.id, HOURLY, held, now=SAVED)
    await pause_sync_policy(owner, owner.id, held, now=SAVED)
    # A paused default holds a connection with an active policy of its own too.
    await save_sync_policy(held_owner, held_owner.id, HOURLY, now=SAVED)
    await save_sync_policy(held_owner, held_owner.id, HOURLY, held_by_default, now=SAVED)
    await pause_sync_policy(held_owner, held_owner.id, now=SAVED)

    started = []

    async def start(credential_id):
        started.append(credential_id)

    # Rows other runs left in a reused database may start too; only ours are judged.
    ours = {inheriting, held, others, held_by_default}
    tick_at = SAVED + timedelta(hours=1)
    await run_due_syncs(tick_at, list_active_credential_ids_for_owner, start)
    assert [credential_id for credential_id in started if credential_id in ours] == [inheriting]

    # The slots are claimed: the same tick again starts nothing of ours.
    started.clear()
    await run_due_syncs(tick_at, list_active_credential_ids_for_owner, start)
    assert [credential_id for credential_id in started if credential_id in ours] == []
