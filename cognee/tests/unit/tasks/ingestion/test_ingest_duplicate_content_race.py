"""Concurrency test for byte-identical add() (gh #4029).

`add()` schedules each data item as its own concurrent ingestion task. Items with
byte-identical content resolve to the same content-derived `data_id`, so the
tasks race to create/link one shared row. Before the fix this crashed — first
with `UNIQUE constraint failed: data.id`, and (after a retry was added) with
`InvalidRequestError: another instance with key ... is already present in this
session` when the retry re-linked `Data` objects loaded in a different session.

The precise failure is timing-dependent (it needs one task to commit between
another task's existence-read and its link write), so this test does not claim to
force it on every run; the fix was validated separately with a stress loop
(unfixed failed ~1-in-5 concurrent adds, fixed passed 40/40). What this test pins
down deterministically is the invariant the fix restores: several concurrent
byte-identical adds must not raise and must collapse to exactly one `Data` row.
To maximize contention it holds every ingestion task at the existence read until
all have arrived before any commits.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_concurrent_identical_files_dedupe_to_single_row(tmp_path, monkeypatch):
    import importlib

    import cognee
    from cognee.infrastructure.databases.relational import (
        create_db_and_tables,
        get_relational_engine,
    )
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.modules.data.models import Data
    from sqlalchemy import func, select

    # The ingestion package re-exports the ingest_data function under the same
    # name as its submodule, so resolve the module explicitly to patch it.
    ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

    monkeypatch.setenv("LLM_API_KEY", "sk-dummy-not-used-by-add")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-dummy-not-used-by-add")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")

    system_dir = tmp_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    cognee.config.system_root_directory(str(system_dir))
    cognee.config.data_root_directory(str(tmp_path / "data"))
    # Drop any relational engine cached by an earlier test so this test's fresh
    # temp-dir database is used, then create its schema. Clear again on the way
    # out so a later test rebuilds its own engine instead of reusing this
    # (now-deleted) temp database.
    create_relational_engine.cache_clear()
    try:
        await create_db_and_tables()

        # add() schedules each item as its own concurrent ingestion task, so
        # several byte-identical files maximize contention on the shared id.
        num_files = 4
        content = "# Well Alpha 42-7\n\nOperator: Example Energy LLC\nStatus: producing\n"
        files = []
        for index in range(num_files):
            path = tmp_path / f"well_alpha_copy_{index}.md"  # distinct names, identical bytes
            path.write_text(content)
            files.append(str(path))

        # Hold every ingestion task at the existence read until all have arrived,
        # so they all see the content-derived data_id as missing before any
        # commits. One-shot: retries (which re-enter this read) skip the barrier
        # so they never deadlock waiting for peers that already passed.
        barrier = asyncio.Barrier(num_files)
        barrier_passed = False
        original_get_dataset_data = ingest_module.get_dataset_data

        async def barriered_get_dataset_data(dataset_id):
            nonlocal barrier_passed
            if not barrier_passed:
                try:
                    await asyncio.wait_for(barrier.wait(), timeout=15)
                except (asyncio.TimeoutError, asyncio.BrokenBarrierError):
                    # Scheduling aligned fewer tasks than expected: don't hang,
                    # just proceed without forcing the interleave.
                    pass
                barrier_passed = True
            return await original_get_dataset_data(dataset_id)

        monkeypatch.setattr(ingest_module, "get_dataset_data", barriered_get_dataset_data)

        # Must not raise (UNIQUE-constraint or InvalidRequestError before the fix).
        await cognee.add(files, dataset_name="dup_race")

        # All files share one content-derived data_id → exactly one Data row.
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            data_count = await session.scalar(select(func.count()).select_from(Data))
        assert data_count == 1, f"expected a single deduplicated Data row, got {data_count}"
    finally:
        create_relational_engine.cache_clear()
