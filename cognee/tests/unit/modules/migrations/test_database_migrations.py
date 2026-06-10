"""Unit tests for the revision-chain migration framework.

These cover the pure chain logic (ordering, head, pending) and the runner's
``_apply`` step with a fake engine and fake migrations — no databases involved.
"""

import asyncio

import pytest

from cognee.modules.migrations.migration import (
    Migration,
    head_revision,
    order_migrations,
    pending_migrations,
    revision_id,
)
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS


async def _noop(engine):
    return None


def _chain() -> list[Migration]:
    """A three-migration chain, intentionally provided out of order."""
    m1 = Migration(slug="m1", cognee_version="1.0.0", up=_noop, down_revision=None)
    m2 = Migration(slug="m2", cognee_version="1.1.0", up=_noop, down_revision=revision_id("m1"))
    m3 = Migration(slug="m3", cognee_version="1.2.0", up=_noop, down_revision=revision_id("m2"))
    return [m3, m1, m2]


def test_revision_id_is_deterministic_and_unique():
    assert revision_id("m1") == revision_id("m1")
    assert revision_id("m1") != revision_id("m2")


def test_order_and_head():
    chain = _chain()
    assert [m.slug for m in order_migrations(chain)] == ["m1", "m2", "m3"]
    assert head_revision(chain) == revision_id("m3")
    assert head_revision([]) is None


def test_pending_none_runs_all():
    assert [m.slug for m in pending_migrations(_chain(), None)] == ["m1", "m2", "m3"]


def test_pending_from_middle():
    assert [m.slug for m in pending_migrations(_chain(), revision_id("m1"))] == ["m2", "m3"]


def test_pending_at_head_runs_nothing():
    assert pending_migrations(_chain(), revision_id("m3")) == []


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


def test_registered_migrations_present_and_skipped_at_head():
    # A version-less database runs the registered migrations...
    assert [m.slug for m in pending_migrations(GRAPH_MIGRATIONS, None)] == [
        "namespace_entity_type_node_ids"
    ]
    assert [m.slug for m in pending_migrations(VECTOR_MIGRATIONS, None)] == [
        "dummy_vector_migration"
    ]
    # ...but a database stamped at head (fresh creation) runs nothing.
    assert pending_migrations(GRAPH_MIGRATIONS, head_revision(GRAPH_MIGRATIONS)) == []
    assert pending_migrations(VECTOR_MIGRATIONS, head_revision(VECTOR_MIGRATIONS)) == []


def test_apply_runs_pending_then_advances_revision():
    from cognee.modules.migrations.runner import _apply

    applied_order: list[str] = []

    def make_up(name):
        async def up(engine):
            applied_order.append(name)

        return up

    m1 = Migration(slug="t1", cognee_version="1.0.0", up=make_up("t1"), down_revision=None)
    m2 = Migration(
        slug="t2", cognee_version="1.1.0", up=make_up("t2"), down_revision=revision_id("t1")
    )
    chain = [m1, m2]
    fake_engine = object()

    applied, new_revision = asyncio.run(_apply(fake_engine, chain, None))
    assert applied == ["t1", "t2"]
    assert new_revision == revision_id("t2")
    assert applied_order == ["t1", "t2"]

    # Re-running from head applies nothing and keeps the stored revision.
    applied_order.clear()
    applied_again, revision_again = asyncio.run(_apply(fake_engine, chain, revision_id("t2")))
    assert applied_again == []
    assert revision_again == revision_id("t2")
    assert applied_order == []


def test_runner_is_noop_without_access_control(monkeypatch):
    """Revisions live on per-dataset dataset_database rows; with access control
    off there are none, so the runner must return without touching anything."""
    import cognee.modules.migrations.runner as runner

    monkeypatch.setattr(runner, "backend_access_control_enabled", lambda: False)

    async def _explode():
        raise AssertionError("must not query dataset databases in OFF mode")

    monkeypatch.setattr(runner, "get_dataset_databases", _explode)

    assert asyncio.run(runner.run_database_migrations()) == []
