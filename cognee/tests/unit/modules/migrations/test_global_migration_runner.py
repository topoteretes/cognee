"""Integration test for the global (access-control-OFF) migration runner.

Exercises ``run_database_migrations`` against real databases with backend
access control forced off, validating the reserved global ``dataset_database``
row: creation, head-stamping on a fresh database, running migrations on a
revision-less (pre-feature) database, idempotency, and that the reserved
dataset is hidden from listings.
"""

import asyncio

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.data.methods.get_datasets import get_datasets
from cognee.modules.data.models import Dataset
from cognee.modules.migrations import runner
from cognee.modules.migrations.constants import GLOBAL_DATASET_ID
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.migration import head_revision
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase


async def _get_global_row():
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return await session.get(DatasetDatabase, GLOBAL_DATASET_ID)


async def _clear_global_revisions():
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await session.get(DatasetDatabase, GLOBAL_DATASET_ID)
        row.graph_migration_revision = None
        row.vector_migration_revision = None
        await session.commit()


async def _cleanup():
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await session.get(DatasetDatabase, GLOBAL_DATASET_ID)
        if row is not None:
            await session.delete(row)
        dataset = await session.get(Dataset, GLOBAL_DATASET_ID)
        if dataset is not None:
            await session.delete(dataset)
        await session.commit()


async def _run():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()
    await _cleanup()  # start from a clean slate

    try:
        # First run creates the reserved global row (anchored to the reserved
        # dataset to satisfy the FK) and leaves it at head. Whether migrations
        # actually run on this first pass depends on whether the global graph is
        # already populated (is_empty), so we only assert the end state here.
        summary = await runner.run_database_migrations()
        assert len(summary) == 1, summary

        row = await _get_global_row()
        assert row is not None
        assert row.dataset_id == GLOBAL_DATASET_ID
        assert row.graph_migration_revision == head_revision(GRAPH_MIGRATIONS)
        assert row.vector_migration_revision == head_revision(VECTOR_MIGRATIONS)
        assert row.cognee_version  # recorded for audit

        # The reserved global dataset must be hidden from normal listings.
        user = await get_default_user()
        listed = await get_datasets(user.id)
        assert GLOBAL_DATASET_ID not in [dataset.id for dataset in listed]

        # Simulate a pre-feature database (no recorded revision) -> every
        # migration runs, then the revision advances to head.
        await _clear_global_revisions()
        summary2 = await runner.run_database_migrations()
        assert summary2[0]["graph_migrations_applied"] == ["namespace_entity_type_node_ids"]
        assert summary2[0]["vector_migrations_applied"] == ["dummy_vector_migration"]
        row2 = await _get_global_row()
        assert row2.graph_migration_revision == head_revision(GRAPH_MIGRATIONS)
        assert row2.vector_migration_revision == head_revision(VECTOR_MIGRATIONS)

        # Idempotent: at head nothing runs.
        summary3 = await runner.run_database_migrations()
        assert summary3[0]["graph_migrations_applied"] == []
        assert summary3[0]["vector_migrations_applied"] == []
    finally:
        await _cleanup()


def test_global_database_migrations(monkeypatch):
    # Force the single-global-database mode regardless of environment config.
    monkeypatch.setattr(runner, "backend_access_control_enabled", lambda: False)
    asyncio.run(_run())


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
