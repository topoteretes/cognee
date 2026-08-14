"""Large-scale migration release test — Phase 2 (runs on the CURRENT branch).

Takes over the legacy-seeded 2×~100k-node system from phase 1 and proves the
release migrates it correctly, mirroring the manual 1.5.0 release validation:

1. ``cognee.run_migrations()`` — alembic + the cognee migration chain over
   both datasets — is TIMED; both datasets must reach the chain head with no
   recorded error. This includes the dataset-scoping fork split and the full
   ``rekey_fork_document_ids`` graph re-key of the fork dataset (the scenario
   that could never finish before COG-6112's chunked/batched fixes).
2. Structural verification: both datasets serve their full graph (node/edge
   counts intact and identical across the two identically-seeded datasets),
   each dataset owns exactly one Data row, and the two rows carry DIFFERENT
   ids (the fork actually split).
3. Functional verification: search executes on the migrated data, and a
   post-migration mock-replay ingest into one dataset completes — the write
   path works on the migrated stores.

Runs with the same replay mocks as phase 1 (zero AI calls, deterministic).
"""

import asyncio
import os
import time
from pathlib import Path

DATASET_A = "large_fork_a"
DATASET_B = "large_fork_b"

MEMORIES_FILE = Path(os.environ["LARGE_MEMORIES_FILE"])
MOCK_FILE = Path(os.environ["LARGE_MOCK_FILE"])

# Scale floors come from the workflow (regular mock: ~3.8k nodes/dataset;
# the 27x large mock: ~100k). Cross-dataset equality is asserted regardless,
# so exact pipeline-era counts never need pinning.
MIN_NODES = int(os.environ.get("MIN_NODES", "3000"))
MIN_EDGES = int(os.environ.get("MIN_EDGES", "3000"))


async def dataset_graph_counts(dataset_name: str):
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.methods import get_datasets_by_name
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    datasets = await get_datasets_by_name([dataset_name], user.id)
    assert datasets, f"dataset {dataset_name} not found after migration"
    dataset = datasets[0]
    async with set_database_global_context_variables(dataset.id, user.id):
        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
        return dataset, len(nodes), len(edges)


async def main():
    from cognee.tests.utils.mock_ingestion import install_mocks, load_mock_data

    # Install replay mocks BEFORE migrations: generic (non-native) vector
    # backends re-embed during re-key, and post-migration verification
    # searches embed queries. MOCK_EMBEDDING must match phase 1's vectors.
    install_mocks(load_mock_data(MOCK_FILE))

    import cognee

    # ── 1. Migrations (timed) ────────────────────────────────────────────
    t0 = time.time()
    failed = await cognee.run_migrations()
    migration_seconds = time.time() - t0
    print(f"phase2: run_migrations completed in {migration_seconds:.1f}s", flush=True)
    assert not failed, f"migrations FAILED for databases: {failed}"

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.migrations.registry import MIGRATIONS
    from cognee.modules.users.models import DatasetDatabase

    chain_head = MIGRATIONS[-1].slug
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(DatasetDatabase))).scalars().all()
        assert rows, "no dataset_database rows after migration"
        for row in rows:
            assert row.migration_last_error in (None, ""), (
                f"dataset {row.dataset_id} failed migration: {row.migration_last_error}"
            )
            assert row.migration_revision == chain_head, (
                f"dataset {row.dataset_id} stuck at {row.migration_revision}, expected {chain_head}"
            )
    print(f"phase2: all datasets at chain head '{chain_head}' with no errors", flush=True)

    # ── 2. Structure: graphs intact, fork actually split ─────────────────
    dataset_a, nodes_a, edges_a = await dataset_graph_counts(DATASET_A)
    dataset_b, nodes_b, edges_b = await dataset_graph_counts(DATASET_B)
    print(f"phase2: {DATASET_A} {nodes_a} nodes / {edges_a} edges", flush=True)
    print(f"phase2: {DATASET_B} {nodes_b} nodes / {edges_b} edges", flush=True)
    assert nodes_a >= MIN_NODES and edges_a >= MIN_EDGES, "dataset A graph lost data"
    assert (nodes_a, edges_a) == (nodes_b, edges_b), (
        "identically-seeded datasets diverged after migration"
    )

    from cognee.modules.data.methods import get_dataset_data

    data_a = await get_dataset_data(dataset_a.id)
    data_b = await get_dataset_data(dataset_b.id)
    assert len(data_a) == 1 and len(data_b) == 1, "expected one document per dataset"
    assert data_a[0].id != data_b[0].id, (
        "fork did not split: both datasets still share one Data row id"
    )
    print(f"phase2: fork split verified ({data_a[0].id} != {data_b[0].id})", flush=True)

    # ── 3. Function: search executes, write path works post-migration ────
    from cognee.api.v1.search import SearchType

    results = await cognee.search(
        query_text="Napoleon marched on Moscow",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET_A],
    )
    assert isinstance(results, list), "search did not execute on migrated data"
    print(f"phase2: CHUNKS search executed ({len(results)} result group(s))", flush=True)

    from cognee.tests.utils.mock_ingestion import load_memories, memory_to_text

    extra_text = memory_to_text(load_memories(MEMORIES_FILE)[0])
    await cognee.add([extra_text], dataset_name=DATASET_A)
    await cognee.cognify(datasets=[DATASET_A])
    _, nodes_after, _ = await dataset_graph_counts(DATASET_A)
    assert nodes_after >= nodes_a, "post-migration ingest lost nodes"
    print(f"phase2: post-migration ingest OK ({nodes_a} -> {nodes_after} nodes)", flush=True)

    print(f"PHASE2 PASSED — migration took {migration_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
