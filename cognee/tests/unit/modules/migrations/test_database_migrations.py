"""Unit tests for the revision-chain migration framework.

These cover the pure chain logic (ordering, head, pending, downgrade spans),
the runner's ``_apply`` step with fakes, the registry invariants (shipped slugs
are immutable, chains validate at import), and the FROZEN id derivations of the
shipped migration (pinned as literal UUIDs — if these ever change, the shipped
revision changed meaning, which is forbidden).
"""

import asyncio
import os

import pytest

from cognee.modules.migrations.migration import (
    Migration,
    head_revision,
    migrations_to_downgrade,
    order_migrations,
    pending_migrations,
)
from cognee.modules.migrations.registry import MIGRATIONS


async def _noop(context):
    return None


def _chain(with_downs: bool = False) -> list[Migration]:
    """A three-migration chain, intentionally provided out of order."""
    down = _noop if with_downs else None
    m1 = Migration(slug="m1", cognee_version="1.0.0", up=_noop, down_revision=None, down=down)
    m2 = Migration(slug="m2", cognee_version="1.1.0", up=_noop, down_revision="m1", down=down)
    m3 = Migration(slug="m3", cognee_version="1.2.0", up=_noop, down_revision="m2", down=down)
    return [m3, m1, m2]


def test_revision_is_the_slug():
    # Stored verbatim so an operator can read a dataset_database row directly.
    assert _chain()[0].revision == "m3"


def test_order_and_head():
    chain = _chain()
    assert [m.slug for m in order_migrations(chain)] == ["m1", "m2", "m3"]
    assert head_revision(chain) == "m3"
    assert head_revision([]) is None


def test_pending_none_runs_all():
    assert [m.slug for m in pending_migrations(_chain(), None)] == ["m1", "m2", "m3"]


def test_pending_from_middle():
    assert [m.slug for m in pending_migrations(_chain(), "m1")] == ["m2", "m3"]


def test_pending_at_head_runs_nothing():
    assert pending_migrations(_chain(), "m3") == []


def test_pending_unknown_revision_runs_nothing_but_warns(caplog):
    # Database is ahead of / diverged from this code -> no-op, but NEVER a
    # silent one: a renamed slug / corrupted revision row looks identical and
    # would otherwise permanently disable migrations with zero diagnostics.
    with caplog.at_level("WARNING", logger="cognee.modules.migrations.migration"):
        assert pending_migrations(_chain(), "some-unknown-revision") == []
    assert any("unknown to this chain" in record.message for record in caplog.records)


def test_branched_chain_raises():
    a = Migration(slug="a", cognee_version="1.0.0", up=_noop, down_revision=None)
    b = Migration(slug="b", cognee_version="1.0.0", up=_noop, down_revision=None)
    with pytest.raises(ValueError):
        order_migrations([a, b])


def test_disconnected_chain_raises():
    a = Migration(slug="a", cognee_version="1.0.0", up=_noop, down_revision=None)
    orphan = Migration(
        slug="orphan", cognee_version="1.0.0", up=_noop, down_revision="missing-parent"
    )
    with pytest.raises(ValueError):
        order_migrations([a, orphan])


def test_pending_with_explicit_target_stops_at_target():
    # alembic-style partial upgrade: `upgrade m2` applies up to and including m2.
    assert [m.slug for m in pending_migrations(_chain(), None, "m2")] == ["m1", "m2"]
    assert [m.slug for m in pending_migrations(_chain(), "m1", "m2")] == ["m2"]
    assert pending_migrations(_chain(), "m2", "m2") == []
    # Stored beyond the target -> nothing to do (use downgrade to go back).
    assert pending_migrations(_chain(), "m3", "m2") == []


def test_pending_unknown_target_raises():
    with pytest.raises(ValueError, match="unknown"):
        pending_migrations(_chain(), None, "mystery")


# ── downgrade spans ──────────────────────────────────────────────────────────


def test_downgrade_full_span_newest_first():
    chain = _chain(with_downs=True)
    assert [m.slug for m in migrations_to_downgrade(chain, "m3", None)] == ["m3", "m2", "m1"]


def test_downgrade_partial_span():
    chain = _chain(with_downs=True)
    assert [m.slug for m in migrations_to_downgrade(chain, "m3", "m1")] == ["m3", "m2"]


def test_downgrade_nothing_applied_is_noop():
    assert migrations_to_downgrade(_chain(with_downs=True), None, None) == []


def test_downgrade_unknown_stored_raises():
    # Unlike pending_migrations, downgrading an unknown state is an error:
    # it is always an explicit operator action, never best-effort.
    with pytest.raises(ValueError, match="unknown"):
        migrations_to_downgrade(_chain(with_downs=True), "mystery", None)


def test_downgrade_target_ahead_of_stored_raises():
    with pytest.raises(ValueError, match="ahead"):
        migrations_to_downgrade(_chain(with_downs=True), "m1", "m3")


def test_downgrade_span_without_down_raises():
    # m2 has no down(): a chain cannot skip a step.
    chain = _chain(with_downs=False)
    with pytest.raises(ValueError, match="without a down"):
        migrations_to_downgrade(chain, "m3", None)


# ── shipped registries: immutable history ────────────────────────────────────


def test_shipped_slugs_are_pinned():
    """These slugs are stored in customer databases. They are APPEND-ONLY:
    if this test fails because a slug changed or vanished, every stamped
    deployment silently stops migrating — fix the chain, not this test."""
    # The vector adapter storage sync is intentionally NOT in this chain — it is
    # version-gated and run by the runner after the chain (see registry.py).
    assert [m.slug for m in order_migrations(MIGRATIONS)] == [
        "namespace_entity_type_node_ids",
        "namespace_edge_type_point_ids",
    ]


def test_registered_migrations_skipped_at_head():
    assert pending_migrations(MIGRATIONS, head_revision(MIGRATIONS)) == []


def test_shipped_migrations_are_downgradable():
    assert all(m.down is not None for m in MIGRATIONS)


def test_frozen_id_derivations_are_pinned():
    """Literal-UUID pins for the shipped migration's FROZEN derivations.

    The migration must mean the same transformation forever; it must NOT track
    live model code. If this fails, someone 'deduplicated' the frozen copies
    against the live functions — revert that, and ship a NEW migration for the
    new scheme instead."""
    from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
        _frozen_bare_id,
        _frozen_model_id,
    )

    assert _frozen_bare_id("alice") == "f7d8be13-2a72-5104-bd02-7a5964737a91"
    assert _frozen_bare_id("entity:alice") == "01ca9835-fc71-5e54-acf1-8de415c20c61"
    assert _frozen_model_id("Entity", "alice") == "afc75e41-4df3-5db9-907e-dcf55750efec"
    assert _frozen_model_id("EntityType", "alice") == "aebfc131-797b-58e8-a145-b892aa7f54a0"


def test_frozen_derivations_currently_match_live_models():
    """Drift alarm (NOT a pin): when the live id scheme diverges from the
    frozen one, a NEW migration is due. If this fails, do not touch the frozen
    copies — append a migration translating frozen-target -> new live scheme."""
    from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
    from cognee.modules.engine.models import Entity, EntityType
    from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
        _frozen_bare_id,
        _frozen_model_id,
    )

    assert _frozen_bare_id("Alice Smith") == str(generate_node_id("Alice Smith"))
    assert _frozen_model_id("Entity", "alice smith") == str(Entity.id_for("alice smith"))
    assert _frozen_model_id("EntityType", "person") == str(EntityType.id_for("person"))


# ── runner pieces ────────────────────────────────────────────────────────────


def test_apply_runs_pending_in_order_and_stamps_per_step(monkeypatch):
    import cognee.modules.migrations.runner as runner

    applied_order: list[str] = []
    stamps: list = []

    def make_up(name):
        async def up(context):
            applied_order.append(name)

        return up

    async def stamp(revision):
        stamps.append(revision)

    m1 = Migration(slug="t1", cognee_version="1.0.0", up=make_up("t1"), down_revision=None)
    m2 = Migration(slug="t2", cognee_version="1.1.0", up=make_up("t2"), down_revision="t1")
    monkeypatch.setattr(runner, "MIGRATIONS", [m1, m2])

    applied = asyncio.run(runner._apply(object(), None, "head", stamp))
    assert applied == ["t1", "t2"]
    assert applied_order == ["t1", "t2"]
    assert stamps == ["t1", "t2"]  # stamped after EACH applied step

    # Re-running from head applies nothing.
    applied_order.clear()
    assert asyncio.run(runner._apply(object(), "t2", "head", stamp)) == []
    assert applied_order == []


def test_revert_span_runs_downs_newest_first_and_stamps_per_step():
    """Per-step stamping: after each down(), the stored revision is moved to
    that migration's down_revision — a failure mid-span can never leave the
    bookkeeping pointing at already-reverted data."""
    from cognee.modules.migrations.migration import migrations_to_downgrade
    from cognee.modules.migrations.runner import _revert_span

    reverted_order: list[str] = []
    stamps: list = []

    def make_down(name):
        async def down(context):
            reverted_order.append(name)

        return down

    async def stamp(revision):
        stamps.append(revision)

    m1 = Migration(
        slug="t1", cognee_version="1.0.0", up=_noop, down_revision=None, down=make_down("t1")
    )
    m2 = Migration(
        slug="t2", cognee_version="1.1.0", up=_noop, down_revision="t1", down=make_down("t2")
    )

    span = migrations_to_downgrade([m1, m2], "t2", None)
    reverted = asyncio.run(_revert_span(object(), span, stamp))
    assert reverted == ["t2", "t1"]
    assert reverted_order == ["t2", "t1"]
    assert stamps == ["t1", None]  # stamped after EACH step, ending at base


def test_revert_span_failure_mid_span_keeps_bookkeeping_consistent():
    """t2 reverts and stamps t1; t1's down raises -> the stored revision must
    remain t1 (last consistent state), never t2-over-reverted-data."""
    import pytest

    from cognee.modules.migrations.migration import migrations_to_downgrade
    from cognee.modules.migrations.runner import _revert_span

    stamps: list = []

    async def stamp(revision):
        stamps.append(revision)

    async def ok_down(context):
        return None

    async def boom_down(context):
        raise RuntimeError("backend down")

    m1 = Migration(slug="t1", cognee_version="1.0.0", up=_noop, down_revision=None, down=boom_down)
    m2 = Migration(slug="t2", cognee_version="1.1.0", up=_noop, down_revision="t1", down=ok_down)

    span = migrations_to_downgrade([m1, m2], "t2", None)
    with pytest.raises(RuntimeError):
        asyncio.run(_revert_span(object(), span, stamp))
    assert stamps == ["t1"]  # t2's revert recorded; nothing claims base


def test_downgrade_span_is_synchronous_validation():
    """_downgrade_span is pure validation called without await everywhere —
    it must be a plain function (regression: declaring it async made every
    downgrade fail with 'coroutine is not iterable')."""
    import inspect

    from cognee.modules.migrations.runner import _downgrade_span

    assert not inspect.iscoroutinefunction(_downgrade_span)
    assert _downgrade_span(None, None) == []  # nothing applied -> nothing to revert


def test_runner_routes_to_global_path_without_access_control(monkeypatch):
    """With access control off there are no per-dataset rows: the runner must
    migrate via the single global_database_version row, never the per-dataset
    iteration."""
    import cognee.modules.migrations.runner as runner

    monkeypatch.setattr(runner, "backend_access_control_enabled", lambda: False)

    async def _read_version():
        return None  # avoid touching a real engine; value is irrelevant here

    monkeypatch.setattr(runner, "_read_deployment_version", _read_version)

    async def _explode():
        raise AssertionError("must not query dataset databases in OFF mode")

    monkeypatch.setattr(runner, "get_dataset_databases", _explode)

    sentinel = [{"database": "global", "graph_migrations_applied": ["x"]}]

    async def _fake_global(current_version, version_changed, target="head"):
        assert current_version
        return sentinel

    monkeypatch.setattr(runner, "_run_global_migrations", _fake_global)

    assert asyncio.run(runner.run_database_migrations()) == sentinel


# ── vector adapter storage sync: version-gated, not chain-gated ───────────────


def test_adapter_storage_sync_runs_on_version_change_even_at_chain_head(monkeypatch, tmp_path):
    """The vector adapter storage sync must run on a Cognee VERSION change even
    when the revision chain is already at head (the bug: as a chain entry it
    would never re-run). Drives the real global runner path against a real
    SQLite engine, with the chain stamped at head and the recorded version
    older than the library."""
    import cognee.modules.migrations.runner as runner
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.modules.migrations.models import (
        GLOBAL_DATABASE_VERSION_ROW_ID,
        GlobalDatabaseVersion,
    )

    async def scenario():
        eng = create_relational_engine(str(tmp_path), "ver.db", "", "", "", "", "sqlite")
        await eng.create_database()
        # Chain already at head, recorded version OLDER than the library version.
        async with eng.get_async_session() as session:
            session.add(
                GlobalDatabaseVersion(
                    id=GLOBAL_DATABASE_VERSION_ROW_ID,
                    cognee_version="0.0.0-old",
                    global_migration_revision=head_revision(MIGRATIONS),
                )
            )
            await session.commit()

        run_migrations = _AsyncCounter()
        fake_vector = type("V", (), {"run_migrations": run_migrations})()

        monkeypatch.setattr(runner, "backend_access_control_enabled", lambda: False)
        monkeypatch.setattr(runner, "get_relational_engine", lambda: eng)
        monkeypatch.setattr(runner, "get_cognee_version", lambda: "9.9.9-new")

        async def _fake_graph():
            return object()

        async def _fake_vector():
            return fake_vector

        monkeypatch.setattr(runner, "get_graph_engine", _fake_graph)
        monkeypatch.setattr(runner, "get_vector_engine", _fake_vector)

        # version changed (0.0.0-old != 9.9.9-new) -> adapter sync runs once,
        # even though the chain is at head (no data migration applied).
        result = await runner.run_database_migrations()
        assert result == [{"database": "global", "migrations_applied": []}]
        assert run_migrations.calls == 1

        # Second pass: version now recorded as current -> no version change ->
        # adapter sync does NOT run again.
        result2 = await runner.run_database_migrations()
        assert result2 == [{"database": "global", "migrations_applied": []}]
        assert run_migrations.calls == 1

        await eng.engine.dispose(close=True)

    asyncio.run(scenario())


class _AsyncCounter:
    def __init__(self):
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        return None


# ── cross-process migration lock (SQLite) ────────────────────────────────────


def test_sqlite_migration_lock_is_a_real_cross_process_file_lock(tmp_path):
    """On SQLite the migrate-then-stamp sequence holds an OS file lock: while a
    process is inside ``_migration_lock``, an independent ``FileLock`` on the
    same path (standing in for a second process / worker) cannot acquire it; it
    frees on exit. The lock file lives next to the database, keyed per migration."""
    import cognee.modules.migrations.runner as runner
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from filelock import FileLock, Timeout

    async def scenario():
        eng = create_relational_engine(str(tmp_path), "lock.db", "", "", "", "", "sqlite")
        key = 7
        path = runner._file_lock_path(eng, key)
        assert path is not None
        assert os.path.dirname(path) == os.path.abspath(str(tmp_path))

        async with runner._migration_lock(eng, key):
            with pytest.raises(Timeout):
                FileLock(path).acquire(timeout=0.2)  # a "second process" is blocked

        freed = FileLock(path)  # released on context exit
        freed.acquire(timeout=0.2)
        assert freed.is_locked
        freed.release()
        await eng.engine.dispose(close=True)

    asyncio.run(scenario())


def test_migration_lock_path_skips_in_memory_sqlite():
    """An in-memory / pathless DB has nothing to coordinate across processes."""
    import cognee.modules.migrations.runner as runner

    fake = type(
        "E", (), {"engine": type("X", (), {"url": type("U", (), {"database": ":memory:"})()})()}
    )()
    assert runner._file_lock_path(fake, 1) is None
