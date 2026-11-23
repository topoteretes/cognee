"""Upgrade stops: a database this release cannot migrate is refused before any step runs.

The stop itself was checked end to end against cognee 1.3.0 databases (Postgres graph
with a document shared by two datasets, refused here and migrated through 1.6.10);
these tests pin the runner wiring and the message without a database.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.migrations import runner, upgrade_stops
from cognee.modules.migrations.migration import MigrationContext, pending_migrations
from cognee.modules.migrations.registry import MIGRATIONS
from cognee.modules.migrations.upgrade_stops import (
    UpgradeStop,
    UpgradeThroughReleaseError,
    check_upgrade_stops,
)

STOP_BEFORE = "rekey_fork_document_ids"


def _stop(applies: bool, calls: list) -> UpgradeStop:
    async def predicate(context):
        calls.append(context)
        return applies

    return UpgradeStop(
        before=STOP_BEFORE, release="1.6.10", reason="Test reason.", applies=predicate
    )


def _pending_from(stored_revision):
    return pending_migrations(MIGRATIONS, stored_revision, "head")


def _revision_before_stop() -> str:
    slugs = [migration.slug for migration in MIGRATIONS]
    return slugs[slugs.index(STOP_BEFORE) - 1]


@pytest.mark.asyncio
async def test_a_database_the_stop_applies_to_is_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(upgrade_stops, "UPGRADE_STOPS", (_stop(True, calls),))
    context = MigrationContext(graph_engine=None, vector_engine=None, dataset_id=uuid4())

    with pytest.raises(UpgradeThroughReleaseError, match="need cognee 1.6.10 first"):
        await check_upgrade_stops(context, _pending_from(_revision_before_stop()))

    assert calls == [context]


@pytest.mark.asyncio
async def test_a_database_past_the_stop_is_not_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(upgrade_stops, "UPGRADE_STOPS", (_stop(True, calls),))
    context = MigrationContext(graph_engine=None, vector_engine=None)

    await check_upgrade_stops(context, _pending_from(STOP_BEFORE))

    assert calls == []


@pytest.mark.asyncio
async def test_a_database_the_stop_does_not_apply_to_migrates(monkeypatch):
    calls = []
    monkeypatch.setattr(upgrade_stops, "UPGRADE_STOPS", (_stop(False, calls),))

    await check_upgrade_stops(MigrationContext(None, None), _pending_from(None))

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_runner_refuses_before_applying_any_step(monkeypatch):
    applied = []

    async def refuse(context, pending):
        raise UpgradeThroughReleaseError(_stop(True, []), context)

    monkeypatch.setattr(runner, "check_upgrade_stops", refuse)
    monkeypatch.setattr(
        runner,
        "pending_migrations",
        lambda *args: [SimpleNamespace(slug="x", revision="x", cognee_version="0", up=applied)],
    )

    with pytest.raises(UpgradeThroughReleaseError):
        await runner._apply(MigrationContext(None, None), None, "head", stamp=None)

    assert applied == []


@pytest.mark.parametrize("dataset_id", [uuid4(), None])
def test_the_stored_error_keeps_the_whole_instruction(dataset_id):
    """The runner stores 500 characters of the error with the database."""
    for stop in upgrade_stops.UPGRADE_STOPS:
        error = UpgradeThroughReleaseError(stop, MigrationContext(None, None, dataset_id))

        stored = runner._error_text(error)

        assert stored.endswith("run `cognee-cli upgrade` again.")
        assert f"cognee=={stop.release}" in stored


@pytest.mark.asyncio
async def test_the_shipped_stop_ignores_a_graph_outside_postgres():
    stop = upgrade_stops.UPGRADE_STOPS[0]
    graph_engine = SimpleNamespace()  # no SQLAlchemy engine: Ladybug, Neo4j, ...

    assert await stop.applies(MigrationContext(graph_engine, None, uuid4())) is False


def test_every_stop_names_a_shipped_migration():
    slugs = {migration.slug for migration in MIGRATIONS}

    assert {stop.before for stop in upgrade_stops.UPGRADE_STOPS} <= slugs
