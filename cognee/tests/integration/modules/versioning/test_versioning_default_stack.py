"""Dataset versioning proof on the REAL default stack (issue #3650, Approach 1).

Runs the run-ledger time-travel surface — snapshot, as-of reads, rollback,
reversible forget + undo — against real Ladybug (graph), real LanceDB (vector,
deterministic hash embeddings), and real SQLite (relational ledger). No LLM
and no network: graph state is seeded through the same ``add_data_points``
provenance-stamping write path cognify uses, with explicit ``PipelineRun``
ledger rows so run completion times are deterministic.

What the fidelity assertions pin down:

- as-of T returns *exactly* the artifact and vector-id sets of the runs
  completed by T (chunk/entity vector ids are graph node ids, so the
  graph-derived visible set is asserted against the vector hits directly);
- rollback to T and undo are exact round-trips, including the shared-ownership
  case (an artifact co-owned by two runs must survive the partial rollback and
  come back with both provenance attachments after undo);
- reversible forget restores byte-equal embeddings (no re-embedding) and the
  full provenance columns;
- the as-of boundary is *documented behavior*: a destructive op after T
  shadows earlier state until it is undone (forward-faithful filter, not
  reconstruction through un-undone deletes).
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.pipelines.models import PipelineContext, PipelineRun, PipelineRunStatus
from cognee.modules.users.methods import get_default_user
from cognee.modules.versioning import (
    VersionOp,
    VersionOpStatus,
    create_snapshot,
    get_graph_as_of,
    get_visible_artifacts_as_of,
    rollback_dataset_to,
    search_chunks_as_of,
    undo_version_op,
)
from cognee.tasks.storage.add_data_points import add_data_points

try:
    import ladybug  # noqa: F401
    import lancedb  # noqa: F401

    HAS_DEFAULT_STACK = True
except ModuleNotFoundError:
    HAS_DEFAULT_STACK = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_DEFAULT_STACK, reason="default stack not installed"),
]


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _hash_vector(text: str, size: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:size]]


@pytest_asyncio.fixture
async def versioning_env(request, tmp_path, monkeypatch):
    """Clean per-test default stack under tmp_path with deterministic embeddings."""
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "ladybug")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")

    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    graph_db_config.set(None)
    vector_db_config.set(None)
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "ladybug",
            "graph_dataset_database_handler": "ladybug",
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_dataset_database_handler": "lancedb",
        }
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    # Deterministic offline embeddings: same text -> same vector, no network.
    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    async def _fake_embed_text(self, texts):
        return [_hash_vector(text) for text in texts]

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _fake_embed_text)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "get_vector_size", lambda self: 8)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


async def _log_completed_run(dataset_id: UUID, pipeline_run_id: UUID, completed_at: datetime):
    """Write the COMPLETED ledger row with a controlled completion time."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # One logical run writes several status rows sharing pipeline_run_id;
        # timeline resolution must dedupe on COMPLETED rows only.
        session.add(
            PipelineRun(
                pipeline_run_id=pipeline_run_id,
                pipeline_name="cognify_pipeline",
                pipeline_id=uuid4(),
                status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                dataset_id=dataset_id,
                run_info={},
                created_at=completed_at - timedelta(seconds=1),
            )
        )
        session.add(
            PipelineRun(
                pipeline_run_id=pipeline_run_id,
                pipeline_name="cognify_pipeline",
                pipeline_id=uuid4(),
                status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                dataset_id=dataset_id,
                run_info={},
                created_at=completed_at,
            )
        )
        await session.commit()


async def _seed_run(dataset, user, data_id, pipeline_run_id, nodes, edges=None):
    """Write nodes/edges through the provenance-stamping cognify write path."""
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        await add_data_points(
            nodes,
            custom_edges=edges or [],
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=pipeline_run_id,
            ),
        )


async def _all_graph_node_ids(graph) -> set[str]:
    rows = await graph.query("MATCH (n:Node) RETURN n.id", {})
    return {row[0] for row in rows}


async def _all_graph_edges(graph) -> set[tuple]:
    rows = await graph.query(
        "MATCH (a:Node)-[r:EDGE]->(b:Node) RETURN a.id, b.id, r.relationship_name", {}
    )
    return {(row[0], row[1], row[2]) for row in rows}


def _normalized_node_state(node_data: dict) -> dict:
    """Comparable snapshot: properties + provenance with order-insensitive refs."""
    return {
        node_id: {
            "type": data.node_type,
            "properties": data.node_properties,
            "source_ref_keys": sorted(data.source_ref_keys),
            "source_dataset_ids": sorted(data.source_dataset_ids),
            "source_run_ids": sorted(data.source_run_ids),
            "source_run_refs": sorted(data.source_run_refs),
        }
        for node_id, data in node_data.items()
    }


async def _full_state(graph, vector, collections: list[str]):
    """Graph rows + provenance + raw vector rows (embeddings included)."""
    node_ids = sorted(await _all_graph_node_ids(graph))
    node_state = _normalized_node_state(await graph.get_node_delete_data(node_ids))

    edge_state = {}
    edges = await _all_graph_edges(graph)
    from cognee.infrastructure.databases.provenance import EdgeIdentity

    edge_data = await graph.get_edge_delete_data(
        [EdgeIdentity(source_id=s, target_id=t, relationship_name=r) for s, t, r in edges]
    )
    for edge, data in edge_data.items():
        edge_state[(edge.source_id, edge.target_id, edge.relationship_name)] = {
            "properties": data.edge_properties,
            "source_ref_keys": sorted(data.source_ref_keys),
            "source_run_refs": sorted(data.source_run_refs),
        }

    vector_state = {}
    for collection in collections:
        rows = await vector.get_raw_rows(collection, node_ids)
        vector_state[collection] = {
            row["id"]: (tuple(row["vector"]), tuple(sorted((row["payload"] or {}).keys())))
            for row in rows
        }

    return node_state, edge_state, vector_state


@pytest_asyncio.fixture
async def two_run_dataset(versioning_env):
    """Shared fixture: dataset with two completed runs and a snapshot between.

    R1 (t1): Alice, Bob + knows-edge for data item 1.
    snapshot "v1" at t_mid.
    R2 (t2): Carol for data item 2, plus a second ownership ref on Bob
             (shared artifact across runs — the partial-rollback case).
    """
    user = await get_default_user()
    dataset = await create_authorized_dataset("versioning_fixture_dataset", user)

    add1 = await cognee.add("Alice knows Bob.", dataset_name=dataset.name, user=user)
    add2 = await cognee.add("Carol appears later.", dataset_name=dataset.name, user=user)
    data_id_1 = add1.data_ingestion_info[0]["data_id"]
    data_id_2 = add2.data_ingestion_info[0]["data_id"]

    base = datetime.now(timezone.utc)
    t1, t_mid, t2 = (
        base - timedelta(minutes=10),
        base - timedelta(minutes=5),
        base - timedelta(minutes=1),
    )

    run1, run2 = uuid4(), uuid4()

    alice, bob, carol = Person(name="Alice"), Person(name="Bob"), Person(name="Carol")

    await _seed_run(
        dataset,
        user,
        data_id_1,
        run1,
        [alice, bob],
        edges=[(str(alice.id), str(bob.id), "knows", {"edge_text": "knows"})],
    )
    await _log_completed_run(dataset.id, run1, t1)

    snapshot = await create_snapshot(dataset.id, "v1", as_of_time=t_mid)

    # R2: new node Carol (data 2) + re-own Bob under data 2 (shared artifact).
    await _seed_run(dataset, user, data_id_2, run2, [carol, Person(id=bob.id, name="Bob")])
    await _log_completed_run(dataset.id, run2, t2)

    return SimpleNamespace(
        user=user,
        dataset=dataset,
        data_id_1=UUID(str(data_id_1)),
        data_id_2=UUID(str(data_id_2)),
        run1=run1,
        run2=run2,
        t1=t1,
        t_mid=t_mid,
        t2=t2,
        snapshot=snapshot,
        alice=alice,
        bob=bob,
        carol=carol,
    )


async def test_as_of_returns_exactly_r1_state(two_run_dataset):
    """as-of the snapshot shows exactly R1's nodes/edges; live view shows R2 too."""
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()

        nodes, edges = await get_graph_as_of(graph, fx.dataset.id, "v1")
        as_of_node_ids = {node["id"] for node in nodes}
        as_of_edges = {
            (edge["source_id"], edge["target_id"], edge["relationship_name"]) for edge in edges
        }

        assert as_of_node_ids == {str(fx.alice.id), str(fx.bob.id)}
        assert as_of_edges == {(str(fx.alice.id), str(fx.bob.id), "knows")}

        live_node_ids = await _all_graph_node_ids(graph)
        assert str(fx.carol.id) in live_node_ids  # live view has R2

        # Datetime as-of works identically to the snapshot name.
        nodes_by_time, _ = await get_graph_as_of(graph, fx.dataset.id, fx.t_mid)
        assert {node["id"] for node in nodes_by_time} == as_of_node_ids

        # After t2, everything is visible.
        nodes_now, _ = await get_graph_as_of(graph, fx.dataset.id, datetime.now(timezone.utc))
        assert {node["id"] for node in nodes_now} == {
            str(fx.alice.id),
            str(fx.bob.id),
            str(fx.carol.id),
        }


async def test_as_of_vector_search_filters_to_exact_id_set(two_run_dataset):
    """The vector post-filter returns exactly the visible vector-id set at T.

    Vector ids in per-type collections are graph node ids — assert the as-of
    hits equal R1's id set, not merely a subset (over- and under-return both
    fail).
    """
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        hits = await search_chunks_as_of(
            graph,
            vector,
            fx.dataset.id,
            "person",
            "v1",
            top_k=10,
            collection_name="Person_name",
        )
        assert {str(hit.id) for hit in hits} == {str(fx.alice.id), str(fx.bob.id)}

        live_hits = await vector.search("Person_name", query_text="person", limit=None)
        assert str(fx.carol.id) in {str(hit.id) for hit in live_hits}


async def test_reversible_forget_then_undo_restores_exactly(two_run_dataset):
    """forget(reversible) -> undo restores graph rows, provenance, and embeddings."""
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        before = await _full_state(graph, vector, ["Person_name"])

    result = await cognee.forget(
        data_id=fx.data_id_1,
        dataset_id=fx.dataset.id,
        memory_only=True,
        reversible=True,
        user=fx.user,
    )
    assert result["status"] == "success"
    operation_id = result["operation_id"]

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        # Alice (owned only by data 1) is gone from graph AND vector store;
        # Bob (shared with data 2) survives with data 1's ref detached.
        remaining_ids = await _all_graph_node_ids(graph)
        assert str(fx.alice.id) not in remaining_ids
        assert str(fx.bob.id) in remaining_ids
        assert await vector.get_raw_rows("Person_name", [str(fx.alice.id)]) == []

        bob_data = await graph.get_node_delete_data([str(fx.bob.id)])
        assert len(bob_data[str(fx.bob.id)].source_ref_keys) == 1

    await cognee.undo(operation_id, dataset_id=fx.dataset.id, user=fx.user)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        after = await _full_state(graph, vector, ["Person_name"])

    assert after == before  # exact restore: properties, provenance, embeddings


async def test_forget_after_t_shadows_as_of_until_undone(two_run_dataset):
    """Documented as-of boundary: a destructive op after T hides pre-T state.

    as-of is a forward-faithful filter over the live store, not reconstruction
    through un-undone deletes: forgetting data 1 (ingested in R1, i.e. before
    the snapshot cut) makes as_of("v1") stop returning Alice. Undoing the
    forget makes the same read exact again.
    """
    fx = two_run_dataset

    result = await cognee.forget(
        data_id=fx.data_id_1,
        dataset_id=fx.dataset.id,
        memory_only=True,
        reversible=True,
        user=fx.user,
    )

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        visible_nodes, _ = await get_visible_artifacts_as_of(graph, fx.dataset.id, "v1")
        # The boundary: Alice existed at T but a post-T forget shadows her.
        assert str(fx.alice.id) not in visible_nodes

    await cognee.undo(result["operation_id"], dataset_id=fx.dataset.id, user=fx.user)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        visible_nodes, _ = await get_visible_artifacts_as_of(graph, fx.dataset.id, "v1")
        assert visible_nodes == {str(fx.alice.id), str(fx.bob.id)}


async def test_rollback_to_snapshot_and_undo_with_shared_ownership(two_run_dataset):
    """rollback reverses post-T runs newest-first; undo restores them exactly.

    Shared-ownership pin: Bob is owned by data 1 (attached in R1) and data 2
    (attached in R2). Rolling back to v1 removes R2's contribution — Carol is
    deleted, Bob *survives* with only R1's ref. Undo restores Carol and Bob's
    second ownership attachment.
    """
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        before = await _full_state(graph, vector, ["Person_name"])

    result = await cognee.rollback("v1", dataset_id=fx.dataset.id, user=fx.user)
    assert result["status"] == "success"
    assert result["rolled_back_runs"] == [str(fx.run2)]

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        remaining_ids = await _all_graph_node_ids(graph)
        assert str(fx.carol.id) not in remaining_ids  # R2-only artifact removed
        assert str(fx.bob.id) in remaining_ids  # co-owned artifact survives

        bob_data = await graph.get_node_delete_data([str(fx.bob.id)])
        assert len(bob_data[str(fx.bob.id)].source_ref_keys) == 1  # R2's ref detached
        assert await vector.get_raw_rows("Person_name", [str(fx.carol.id)]) == []

        # The rolled-back state matches the as-of view of the same cut.
        visible_nodes, _ = await get_visible_artifacts_as_of(graph, fx.dataset.id, "v1")
        assert visible_nodes == {str(fx.alice.id), str(fx.bob.id)}
        assert visible_nodes <= remaining_ids

    await cognee.undo(result["operation_id"], dataset_id=fx.dataset.id, user=fx.user)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        after = await _full_state(graph, vector, ["Person_name"])

    assert after == before


async def test_rollback_noop_when_nothing_after_t(two_run_dataset):
    fx = two_run_dataset

    result = await cognee.rollback(
        datetime.now(timezone.utc), dataset_id=fx.dataset.id, user=fx.user
    )
    assert result == {"operation_id": None, "rolled_back_runs": [], "status": "noop"}

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        assert {str(fx.alice.id), str(fx.bob.id), str(fx.carol.id)} <= (
            await _all_graph_node_ids(graph)
        )


async def test_reversible_forget_fails_closed_on_unsupported_backend(two_run_dataset, monkeypatch):
    """Backends without graph provenance raise before any data is touched."""
    fx = two_run_dataset

    from cognee.modules.versioning.methods import operations as operations_module

    async def _not_in_graph(_graph_engine):
        return False

    monkeypatch.setattr(operations_module, "stores_provenance_in_graph", _not_in_graph)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        before = await _all_graph_node_ids(graph)

    with pytest.raises(UnsupportedProvenanceCapability):
        await cognee.forget(
            data_id=fx.data_id_1,
            dataset_id=fx.dataset.id,
            memory_only=True,
            reversible=True,
            user=fx.user,
        )

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        assert await _all_graph_node_ids(graph) == before  # nothing was deleted


async def test_undo_refuses_outside_retention_window(two_run_dataset):
    fx = two_run_dataset

    result = await cognee.forget(
        data_id=fx.data_id_1,
        dataset_id=fx.dataset.id,
        memory_only=True,
        reversible=True,
        user=fx.user,
    )
    op_id = UUID(result["operation_id"])

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        op = (await session.execute(select(VersionOp).where(VersionOp.id == op_id))).scalars().one()
        op.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        await session.commit()

    with pytest.raises(ValueError, match="retention window"):
        await cognee.undo(op_id, dataset_id=fx.dataset.id, user=fx.user)


async def test_undo_twice_refuses_and_ledger_tracks_status(two_run_dataset):
    fx = two_run_dataset

    result = await cognee.forget(
        data_id=fx.data_id_1,
        dataset_id=fx.dataset.id,
        memory_only=True,
        reversible=True,
        user=fx.user,
    )
    op_id = UUID(result["operation_id"])

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        op = (await session.execute(select(VersionOp).where(VersionOp.id == op_id))).scalars().one()
        assert op.status == VersionOpStatus.APPLIED

    await cognee.undo(op_id, dataset_id=fx.dataset.id, user=fx.user)

    async with db_engine.get_async_session() as session:
        op = (await session.execute(select(VersionOp).where(VersionOp.id == op_id))).scalars().one()
        assert op.status == VersionOpStatus.UNDONE

    with pytest.raises(ValueError, match="already been undone"):
        await cognee.undo(op_id, dataset_id=fx.dataset.id, user=fx.user)


async def test_undo_is_idempotent_after_partial_restore(two_run_dataset):
    """Crash-recovery contract: replaying a restore converges to the same state.

    Restore primitives are MERGE upserts / set-merge attaches, so re-running
    the inverse over an already-restored store must not duplicate artifacts or
    provenance attachments.
    """
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        before = await _full_state(graph, vector, ["Person_name"])

    result = await cognee.forget(
        data_id=fx.data_id_1,
        dataset_id=fx.dataset.id,
        memory_only=True,
        reversible=True,
        user=fx.user,
    )
    op_id = UUID(result["operation_id"])

    from cognee.modules.versioning import get_version_op
    from cognee.modules.versioning.methods.inverse import restore_inverse_step

    op = await get_version_op(op_id)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        # Simulate a crash mid-undo: the first replay happens, then the whole
        # undo runs again from the top.
        await restore_inverse_step(graph, vector, op.payload["steps"][0])

    await undo_version_op(op_id)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()

        after = await _full_state(graph, vector, ["Person_name"])

    assert after == before


async def test_concurrent_writer_during_reversible_forget(two_run_dataset):
    """A concurrent cognify-style writer must not corrupt forget/undo.

    Reversible forget of data 1 races a writer adding artifacts for data 2.
    Neither operation may fail (no 'database is locked' regressions), the
    forgotten artifacts must be restorable, and the concurrent write must
    survive both the forget and the undo untouched.
    """
    fx = two_run_dataset

    concurrent_nodes = [Person(name=f"Concurrent-{i}") for i in range(5)]
    run3 = uuid4()

    async def _concurrent_writes():
        for node in concurrent_nodes:
            await _seed_run(fx.dataset, fx.user, fx.data_id_2, run3, [node])

    async def _reversible_forget():
        return await cognee.forget(
            data_id=fx.data_id_1,
            dataset_id=fx.dataset.id,
            memory_only=True,
            reversible=True,
            user=fx.user,
        )

    result, _ = await asyncio.gather(_reversible_forget(), _concurrent_writes())
    assert result["status"] == "success"

    await cognee.undo(result["operation_id"], dataset_id=fx.dataset.id, user=fx.user)

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()

        node_ids = await _all_graph_node_ids(graph)
        # Forgotten-and-undone artifacts are back.
        assert str(fx.alice.id) in node_ids
        # Concurrent writes survived forget + undo.
        assert {str(node.id) for node in concurrent_nodes} <= node_ids


async def test_snapshot_names_are_unique_per_dataset(two_run_dataset):
    fx = two_run_dataset

    with pytest.raises(ValueError, match="already exists"):
        await create_snapshot(fx.dataset.id, "v1")

    listed = await cognee.list_snapshots(dataset_id=fx.dataset.id, user=fx.user)
    assert [item["name"] for item in listed] == ["v1"]


async def test_as_of_before_first_run_is_empty(two_run_dataset):
    fx = two_run_dataset

    async with set_database_global_context_variables(fx.dataset.id, fx.dataset.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await get_graph_as_of(graph, fx.dataset.id, fx.t1 - timedelta(minutes=5))
    assert nodes == []
    assert edges == []


async def test_rollback_multiple_runs_newest_first_full_round_trip(versioning_env):
    """Three runs, rollback to before R2 (drops R2+R3 newest-first), then undo.

    The R2 artifact is also co-owned by R3 (attach of a second data item's
    ref), so the newest-first order matters: R3's rollback detaches, R2's
    rollback deletes. Undo restores in reverse application order.
    """
    user = await get_default_user()
    dataset = await create_authorized_dataset("versioning_multirun_dataset", user)

    add1 = await cognee.add("first", dataset_name=dataset.name, user=user)
    add2 = await cognee.add("second", dataset_name=dataset.name, user=user)
    add3 = await cognee.add("third", dataset_name=dataset.name, user=user)
    data_ids = [UUID(str(entry.data_ingestion_info[0]["data_id"])) for entry in (add1, add2, add3)]

    base = datetime.now(timezone.utc)
    times = [base - timedelta(minutes=m) for m in (9, 6, 3)]
    runs = [uuid4(), uuid4(), uuid4()]

    n1, n2, n3 = Person(name="One"), Person(name="Two"), Person(name="Three")

    await _seed_run(dataset, user, data_ids[0], runs[0], [n1])
    await _log_completed_run(dataset.id, runs[0], times[0])

    await _seed_run(dataset, user, data_ids[1], runs[1], [n2])
    await _log_completed_run(dataset.id, runs[1], times[1])

    # R3: own node three AND co-own node two (shared across R2/R3).
    await _seed_run(dataset, user, data_ids[2], runs[2], [n3, Person(id=n2.id, name="Two")])
    await _log_completed_run(dataset.id, runs[2], times[2])

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        before = await _full_state(graph, vector, ["Person_name"])

    cut = times[0] + timedelta(minutes=1)  # after R1, before R2
    result = await cognee.rollback(cut, dataset_id=dataset.id, user=user)
    assert result["rolled_back_runs"] == [str(runs[2]), str(runs[1])]  # newest first

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        node_ids = await _all_graph_node_ids(graph)
        assert str(n1.id) in node_ids
        assert str(n2.id) not in node_ids
        assert str(n3.id) not in node_ids

    await cognee.undo(result["operation_id"], dataset_id=dataset.id, user=user)

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        after = await _full_state(graph, vector, ["Person_name"])

    assert after == before
