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
creates it. It also asserts DB input == output across the roundtrip: the
restored graph carries the exact source node ids and edge count (cognee
archives preserve source UUIDs verbatim), and the social layer exported via
``include_permissions=True`` is restored — the archived owner owns the
dataset, granted users exist with their original credentials, and their
grants hold.
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


async def _graph_snapshot(dataset_name, user):
    """(node id set, edge count) of the dataset's graph — the exact-copy key."""
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.methods import get_authorized_existing_datasets

    dataset = (await get_authorized_existing_datasets([dataset_name], "read", user))[0]
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
    real_edges = [e for e in edges if not (e[2] == "SELF" and e[0] == e[1])]
    return {str(node_id) for node_id, _ in nodes}, len(real_edges)


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

        # Social layer: a second user with a granted read, exported alongside
        # the knowledge so the target can restore accounts and grants.
        from cognee.modules.data.methods import get_authorized_existing_datasets
        from cognee.modules.users.methods import create_user, get_default_user
        from cognee.modules.users.permissions.methods import give_permission_on_dataset

        from cognee.modules.users.methods import get_user_by_email

        owner = await get_default_user()
        await create_user("reviewer@example.com", "reviewer-pw")
        reviewer = await get_user_by_email("reviewer@example.com")
        source_dataset = (await get_authorized_existing_datasets([dataset_name], "read", owner))[0]
        await give_permission_on_dataset(reviewer, source_dataset.id, "read")
        reviewer_hash = reviewer.hashed_password

        source_node_ids, source_edge_count = await _graph_snapshot(dataset_name, owner)

        export_result = await cognee.export(
            dataset_name, format="cogx", destination=archive_dir, include_permissions=True
        )
        assert export_result.num_nodes > 0, "Export produced an empty archive (0 nodes)."
        assert export_result.num_edges > 0, "Export produced an archive with 0 edges."
        assert (archive_dir / "permissions.json").exists(), (
            "include_permissions=True did not write the social layer."
        )

        # A second, DEFAULT export (no social layer) for phase 4: the
        # knowledge-only contract must keep working exactly as before.
        plain_archive_dir = pathlib.Path(temporary_directory) / "hotel_cogx_plain"
        await cognee.export(dataset_name, format="cogx", destination=plain_archive_dir)
        assert not (plain_archive_dir / "permissions.json").exists(), (
            "Default export must not write the social layer."
        )

        # The manifest must carry the source store's migration revision (this
        # store is freshly migrated, so: head) — the import uses it to avoid
        # claiming a newer revision than the exported data actually has.
        from cognee.modules.migration.cogx import read_manifest
        from cognee.modules.migrations.migration import head_revision
        from cognee.modules.migrations.registry import MIGRATIONS

        manifest = read_manifest(archive_dir)
        assert manifest.migration_revision == head_revision(MIGRATIONS), (
            f"Archive manifest carries migration revision {manifest.migration_revision!r}, "
            f"expected the source store's stamp {head_revision(MIGRATIONS)!r}."
        )
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

        # --- Phase 3a: DB input == output (exact copy + social layer). ------
        restored_owner = await get_user_by_email(owner.email)
        assert restored_owner is not None, "Archived owner was not restored."
        target_dataset = (
            await get_authorized_existing_datasets([dataset_name], "read", restored_owner)
        )[0]
        assert target_dataset.owner_id == restored_owner.id, (
            "Imported dataset is not owned by the archived owner."
        )

        restored_reviewer = await get_user_by_email("reviewer@example.com")
        assert restored_reviewer is not None, "Granted user was not restored."
        assert restored_reviewer.hashed_password == reviewer_hash, (
            "Restored user's credentials differ from the source."
        )
        # Shared (non-owned) datasets don't resolve by NAME — list the
        # reviewer's readable datasets instead (same asymmetry the
        # permissions demo documents).
        from cognee.modules.users.permissions.methods import get_all_user_permission_datasets

        reviewer_readable = await get_all_user_permission_datasets(restored_reviewer, "read")
        assert any(dataset.id == target_dataset.id for dataset in reviewer_readable), (
            "Restored reviewer lost the granted read permission."
        )

        target_node_ids, target_edge_count = await _graph_snapshot(dataset_name, restored_owner)
        assert target_node_ids == source_node_ids, (
            f"Graph node ids differ after roundtrip: {len(source_node_ids)} source vs "
            f"{len(target_node_ids)} target."
        )
        assert target_edge_count == source_edge_count, (
            f"Graph edge count differs after roundtrip: {source_edge_count} -> {target_edge_count}"
        )

        # --- Phase 3b: the restored graph answers searches. ------------------
        restored_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=QUERY, datasets=[dataset_name]
        )
        assert len(restored_results) != 0, "Search on the restored instance returned nothing."
        logger.info("Restored search results: %s", restored_results)

        # Chunk-level retrieval needs the imported DocumentChunk text to be
        # vector-indexed (regression: rehydrated raw nodes lost their
        # index_fields, so restored archives had no DocumentChunk_text
        # collection and chunk/hybrid retrieval failed with NoDataError).
        chunk_results = await cognee.search(
            query_type=SearchType.CHUNKS, query_text="rooftop observatory", datasets=[dataset_name]
        )
        assert len(chunk_results) != 0, (
            "Chunk search on the restored instance returned nothing — imported "
            "DocumentChunk text was not vector-indexed."
        )

        # --- Phase 4: DEFAULT (knowledge-only) archive keeps the old contract.
        # No social layer: the importing user owns everything, no accounts are
        # created, and the knowledge still round-trips exactly.
        await _wipe_store()
        migrations_startup._startup_migrations_done = False

        plain_result = await cognee.remember(
            COGXArchiveSource(plain_archive_dir), dataset_name=dataset_name
        )
        assert plain_result.status == "completed"

        assert await get_user_by_email("reviewer@example.com") is None, (
            "Knowledge-only import must not create archived users."
        )
        importing_user = await get_default_user()
        plain_dataset = (
            await get_authorized_existing_datasets([dataset_name], "read", importing_user)
        )[0]
        assert plain_dataset.owner_id == importing_user.id, (
            "Knowledge-only import must tie the dataset to the importing user."
        )
        plain_node_ids, plain_edge_count = await _graph_snapshot(dataset_name, importing_user)
        assert plain_node_ids == source_node_ids, (
            "Knowledge-only import no longer round-trips the exact graph."
        )
        assert plain_edge_count == source_edge_count

        print("COGX roundtrip e2e passed: export -> fresh store -> import -> search.")


if __name__ == "__main__":
    asyncio.run(main())
