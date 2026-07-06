"""E2E: export a cognified dataset as a COGX archive, restore it into a fresh
store, and search the restored knowledge graph.

This is the Cognee-to-Cognee instance-migration path (the local half of
push/pull): ``cognee.export(format="cogx")`` on the source machine,
``cognee.remember(COGXArchiveSource(...))`` on the target. The test covers
the full system contract:

- the preserve-mode import survives the pipeline context adaptation
  (regression: ``ctx.pipeline_run_id`` was dropped, crashing every import),
- the import stamps the fresh store's migration revision at head BEFORE the
  rows arrive (regression: an unstamped store replayed the entire
  data-migration chain on the first server start after a restore),
- the restored graph is populated and answerable via search.

The test runs in multi-user mode (backend access control enabled — pinned
below): each dataset_database row must be head-stamped when the import
creates it.
"""

import asyncio
import os
import pathlib
import tempfile

import cognee
from cognee.modules.migration.sources.cogx_archive import COGXArchiveSource
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()

TEXT = """The Aurora Skyline Hotel is a boutique hotel in Berlin founded by Clara Novak in 2015.
The hotel is known for its rooftop observatory and partners with the Berlin Astronomy Club.
Clara Novak previously managed the Lindenhof Resort in Munich before moving to Berlin.
"""

QUERY = "Who founded the Aurora Skyline Hotel and what is the hotel known for?"


async def _wipe_store():
    """Reset to an empty store, as on a machine that never ran Cognee."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _assert_revision_at_head():
    """The restored store's data-migration bookkeeping must be stamped at head.

    Without the stamp the store is indistinguishable from a pre-migration one,
    and the first migration-aware startup replays the whole chain over the
    imported data. In multi-user mode the stamp lives on the per-dataset
    ``dataset_database`` rows, head-stamped when the import creates them.
    """
    from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
    from cognee.modules.migrations.migration import head_revision
    from cognee.modules.migrations.registry import MIGRATIONS

    head = head_revision(MIGRATIONS)

    rows = await get_dataset_databases()
    assert rows, "Import created no dataset_database bookkeeping row for the fresh store."
    for row in rows:
        assert row.migration_revision == head, (
            f"Imported dataset {row.dataset_id} is stamped at {row.migration_revision!r}, "
            f"expected head {head!r} — the first startup would replay the migration chain."
        )


async def main():
    # Disable session-turn gating so searches return direct retrieval results
    # (same rationale as test_library.py).
    os.environ["AUTO_FEEDBACK"] = "False"

    # This test runs in multi-user mode: the stamp assertion reads the
    # per-dataset bookkeeping rows that mode maintains. The flag is read
    # dynamically, so setting it here wins over any ambient .env.
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_cogx_roundtrip")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_cogx_roundtrip")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    dataset_name = "hotel_knowledge"

    with tempfile.TemporaryDirectory() as temporary_directory:
        # The archive must live OUTSIDE the data/system roots so wiping the
        # store between phases cannot delete it (it is the "file shipped to
        # the other machine").
        archive_dir = pathlib.Path(temporary_directory) / "hotel_cogx"

        # --- Phase 1: source instance — build a graph and export it. -------
        await _wipe_store()

        await cognee.add([TEXT], dataset_name)
        await cognee.cognify([dataset_name])

        source_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=QUERY, datasets=[dataset_name]
        )
        assert len(source_results) != 0, "Search on the source instance returned nothing."
        logger.info("Source search results: %s", source_results)

        export_result = await cognee.export(dataset_name, format="cogx", destination=archive_dir)
        assert export_result.num_nodes > 0, "Export produced an empty archive (0 nodes)."
        assert export_result.num_edges > 0, "Export produced an archive with 0 edges."
        logger.info(
            "Exported %d nodes, %d edges to %s",
            export_result.num_nodes,
            export_result.num_edges,
            export_result.destination,
        )

        # --- Phase 2: target instance — restore into a fresh store. --------
        await _wipe_store()

        # run_migrations is once-per-process; on a real restore the import
        # runs in a brand-new process on the target machine. Reset the guard
        # so the import's migration gate treats the wiped store as exactly
        # that: a fresh instance.
        import cognee.modules.migrations.startup as migrations_startup

        migrations_startup._startup_migrations_done = False

        import_result = await cognee.remember(
            COGXArchiveSource(archive_dir), dataset_name=dataset_name
        )
        assert import_result.status == "completed", (
            f"Import did not complete: {import_result.status!r}"
        )
        assert import_result.items_processed > 0, "Import processed nothing."

        (import_summary,) = [
            item for item in import_result.items if item.get("kind") == "migration_import"
        ]
        assert import_summary["graph_nodes"] > 0, "Import restored no graph nodes."
        assert import_summary["graph_edges"] > 0, "Import restored no graph edges."
        logger.info("Import summary: %s", import_summary)

        await _assert_revision_at_head()

        # --- Phase 3: the restored graph answers searches. ------------------
        restored_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=QUERY, datasets=[dataset_name]
        )
        assert len(restored_results) != 0, "Search on the restored instance returned nothing."
        logger.info("Restored search results: %s", restored_results)

        print("COGX roundtrip e2e passed: export -> fresh store -> import -> search.")


if __name__ == "__main__":
    asyncio.run(main())
