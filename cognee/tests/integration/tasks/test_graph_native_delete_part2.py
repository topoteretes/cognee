"""Part 2 default-stack integration gate for graph-native delete/rollback.

These are the REAL end-to-end tests for graph-native delete/rollback as they
WILL run once Part 1 (Ladybug provenance primitives + LanceDB vector deletes)
lands. They drive the public surface — ``cognee.add`` + ``cognee.cognify`` to
build a graph, then ``cognee.forget`` / dataset delete / ``cognify_rollback_handler``
to remove parts of it — and assert the resulting graph + vector state against
the default stack (Ladybug graph + LanceDB vector + SQLite relational).

The whole module is intentionally collected-but-skipped: the routing authority
(``ensure_graph_native_for_new_graph`` / ``is_graph_native_graph``), the
``UnifiedStoreEngine`` orchestration, and the Part 0 contract are all committed,
but the live add/delete/rollback paths do not yet call the unified boundary, and
the Ladybug/LanceDB provenance read+delete primitives are not yet implemented.
Enabling these before Part 1 would exercise the raising defaults on
``GraphDBInterface``/``GraphVectorStoreInterface`` and fail spuriously.

Acceptance bar these encode (all against the default stack, both access-control
modes):
- shared-artifact survival on data-item delete (a node/edge shared between two
  data items keeps its other source ref and stays in the graph + vector store);
- cross-dataset preservation on dataset delete (a node shared across two
  datasets survives deletion of one dataset);
- unowned-edge deletion (an edge whose last source ref is removed is hard-deleted
  from the graph and its EdgeType/Triplet vectors are cleaned);
- rollback removing only run-introduced artifacts (a failed/rolled-back run
  leaves a prior run's artifacts intact);
- retry convergence after an injected vector failure (a vector-delete failure
  leaves graph provenance untouched; the retry converges to the clean state).
"""

from __future__ import annotations

import importlib
import pathlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.unified.get_unified_engine import get_unified_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.graph.provenance import make_source_ref
from cognee.modules.graph.provenance.markers import is_graph_native_graph
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tests.utils.assert_graph_nodes_not_present import assert_graph_nodes_not_present
from cognee.tests.utils.assert_graph_nodes_present import assert_graph_nodes_present
from cognee.tests.utils.assert_nodes_vector_index_not_present import (
    assert_nodes_vector_index_not_present,
)
from cognee.tests.utils.assert_nodes_vector_index_present import assert_nodes_vector_index_present

# Part 2 integration gate: enable once Part 1 real adapters (Ladybug provenance
# primitives + LanceDB) land. Until then these collect but skip.
pytestmark = pytest.mark.skip(
    reason=(
        "Part 2 integration gate: enable once Part 1 real adapters "
        "(Ladybug provenance primitives + LanceDB) land — COG-5522 Part 1"
    )
)


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture(params=[False, True], ids=["no_access_control", "access_control"])
async def graph_native_environment(request, tmp_path, monkeypatch):
    """Clean default-stack environment, parametrized over both access-control modes.

    Mirrors test_cognify_rollback_recovery's fixture (Ladybug + LanceDB + SQLite,
    pruned per test) but additionally toggles ENABLE_BACKEND_ACCESS_CONTROL so the
    same delete/rollback assertions are proven in both single-tenant and
    per-user/dataset-isolated modes. Yields the access-control flag so tests that
    need per-mode behaviour (e.g. multi-user datasets) can branch.
    """
    pytest.importorskip("ladybug")
    pytest.importorskip("lancedb")

    access_control_enabled: bool = request.param

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv(
        "ENABLE_BACKEND_ACCESS_CONTROL", "true" if access_control_enabled else "false"
    )
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "ladybug")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "false")

    test_id = request.node.name
    root = pathlib.Path(tmp_path) / test_id
    system_directory_path = str(root / "system")
    data_directory_path = str(root / "data")

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine

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
    cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield SimpleNamespace(access_control_enabled=access_control_enabled)

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@asynccontextmanager
async def _dataset_context(dataset_id, owner_id):
    async with set_database_global_context_variables(dataset_id, owner_id):
        yield


async def _seed_shared_nodes(dataset, user, data_id, pipeline_run_id, nodes, custom_edges):
    """Write nodes/edges through the real add_data_points so graph-native refs attach.

    Once Part 1+2 land, add_data_points marks the (empty) graph as graph-native
    and stamps source refs / source-run refs / dataset ids directly onto the
    nodes/edges and their vector payloads — no relational nodes/edges ledger
    rows. These helpers stand in for the entity-extraction output of cognify so
    a test can control exactly which artifacts are shared between data items.
    """
    async with _dataset_context(dataset.id, dataset.owner_id):
        await add_data_points(
            nodes,
            custom_edges=custom_edges,
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=pipeline_run_id,
            ),
        )


async def _node_source_refs(node_id):
    """Read the source refs stamped on a graph node via the Part 1 read primitive."""
    graph_engine = await get_graph_engine()
    node = await graph_engine.get_node(str(node_id))
    return set(node.get("source_refs", [])) if node else set()


# ---------------------------------------------------------------------------
# shared-artifact survival on data-item delete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_data_item_delete_keeps_shared_artifacts(graph_native_environment):
    """Deleting one data item drops its source ref but keeps artifacts shared with
    a second data item, in both graph and vector stores."""
    user = await get_default_user()
    dataset = await create_authorized_dataset("gn_shared_artifact", user)

    add_a = await cognee.add("Alice works with Bob.", dataset_name=dataset.name, user=user)
    add_b = await cognee.add("Bob mentors Carol.", dataset_name=dataset.name, user=user)
    data_id_a = add_a.data_ingestion_info[0]["data_id"]
    data_id_b = add_b.data_ingestion_info[0]["data_id"]

    run_a = uuid4()
    run_b = uuid4()

    # Bob is shared across both data items; Alice belongs only to data item A.
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    carol = Person(name="Carol")

    await _seed_shared_nodes(
        dataset,
        user,
        data_id_a,
        run_a,
        [alice, bob],
        [(alice.id, bob.id, "works_with", {"edge_text": "works with"})],
    )
    await _seed_shared_nodes(
        dataset,
        user,
        data_id_b,
        run_b,
        [bob, carol],
        [(bob.id, carol.id, "mentors", {"edge_text": "mentors"})],
    )

    # The freshly created default-stack graph must be graph-native (Part 2 marker).
    assert await is_graph_native_graph(await get_graph_engine())

    await assert_graph_nodes_present([alice, bob, carol])
    await assert_nodes_vector_index_present([alice, bob, carol])

    ref_a = make_source_ref(dataset.id, data_id_a)
    ref_b = make_source_ref(dataset.id, data_id_b)
    assert ref_a in await _node_source_refs(bob.id)
    assert ref_b in await _node_source_refs(bob.id)

    # Delete data item A (memory + record). Bob survives via data item B's ref.
    async with _dataset_context(dataset.id, dataset.owner_id):
        await cognee.forget(data_id=data_id_a, dataset_id=dataset.id, user=user)

    await assert_graph_nodes_not_present([alice])  # solely owned by A → gone
    await assert_graph_nodes_present([bob, carol])  # shared / B-only → kept
    await assert_nodes_vector_index_not_present([alice])
    await assert_nodes_vector_index_present([bob, carol])

    # Bob keeps only data item B's source ref.
    bob_refs = await _node_source_refs(bob.id)
    assert ref_a not in bob_refs
    assert ref_b in bob_refs


# ---------------------------------------------------------------------------
# cross-dataset preservation on dataset delete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dataset_delete_preserves_cross_dataset_artifacts(graph_native_environment):
    """Deleting one dataset keeps a node that also belongs to a second dataset.

    In access-control mode datasets are isolated, so a node genuinely shared
    across two datasets only arises in the shared single-tenant graph; in that
    mode we assert dataset A's solely-owned node is removed while the shared
    backend graph keeps the cross-dataset node. In single-tenant mode both
    datasets live in the same graph and the cross-dataset node must survive."""
    user = await get_default_user()
    dataset_a = await create_authorized_dataset("gn_cross_a", user)
    dataset_b = await create_authorized_dataset("gn_cross_b", user)

    add_a = await cognee.add("Quantum entanglement basics.", dataset_name=dataset_a.name, user=user)
    add_b = await cognee.add("Quantum computing primer.", dataset_name=dataset_b.name, user=user)
    data_id_a = add_a.data_ingestion_info[0]["data_id"]
    data_id_b = add_b.data_ingestion_info[0]["data_id"]

    run_a = uuid4()
    run_b = uuid4()

    a_only = Person(name="Entanglement")
    shared = Person(name="Quantum")  # belongs to both datasets
    b_only = Person(name="Computing")

    await _seed_shared_nodes(
        dataset_a,
        user,
        data_id_a,
        run_a,
        [a_only, shared],
        [(a_only.id, shared.id, "is_about", {"edge_text": "is about"})],
    )
    await _seed_shared_nodes(
        dataset_b,
        user,
        data_id_b,
        run_b,
        [shared, b_only],
        [(shared.id, b_only.id, "is_about", {"edge_text": "is about"})],
    )

    # Delete the whole of dataset A (graph + vector + records).
    await cognee.forget(dataset_id=dataset_a.id, user=user)

    if graph_native_environment.access_control_enabled:
        # Dataset B has its own isolated graph; its artifacts must remain.
        async with _dataset_context(dataset_b.id, dataset_b.owner_id):
            await assert_graph_nodes_present([b_only, shared])
            await assert_nodes_vector_index_present([b_only, shared])
        # Dataset A's isolated graph is gone entirely.
        async with _dataset_context(dataset_a.id, dataset_a.owner_id):
            await assert_graph_nodes_not_present([a_only])
    else:
        # Single shared graph: A-only removed, cross-dataset node + B kept.
        await assert_graph_nodes_not_present([a_only])
        await assert_graph_nodes_present([shared, b_only])
        await assert_nodes_vector_index_not_present([a_only])
        await assert_nodes_vector_index_present([shared, b_only])
        # The shared node now belongs only to dataset B.
        graph_engine = await get_graph_engine()
        shared_node = await graph_engine.get_node(str(shared.id))
        assert str(dataset_a.id) not in set(shared_node.get("dataset_ids", []))
        assert str(dataset_b.id) in set(shared_node.get("dataset_ids", []))


# ---------------------------------------------------------------------------
# unowned-edge deletion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_data_item_delete_hard_deletes_unowned_edge(graph_native_environment):
    """An edge whose last source ref is removed is hard-deleted from the graph and
    its EdgeType/Triplet vectors are cleaned, even when its endpoint nodes survive."""
    user = await get_default_user()
    dataset = await create_authorized_dataset("gn_unowned_edge", user)

    add_a = await cognee.add("Alice collaborates with Bob.", dataset_name=dataset.name, user=user)
    add_b = await cognee.add("Alice and Bob both exist.", dataset_name=dataset.name, user=user)
    data_id_a = add_a.data_ingestion_info[0]["data_id"]
    data_id_b = add_b.data_ingestion_info[0]["data_id"]

    alice = Person(name="Alice")
    bob = Person(name="Bob")

    # The collaborates_with edge is introduced only by data item A; both nodes
    # are kept alive by data item B (no edge), so deleting A must drop the edge
    # while keeping both endpoint nodes.
    await _seed_shared_nodes(
        dataset,
        user,
        data_id_a,
        uuid4(),
        [alice, bob],
        [(alice.id, bob.id, "collaborates_with", {"edge_text": "collaborates with"})],
    )
    await _seed_shared_nodes(
        dataset,
        user,
        data_id_b,
        uuid4(),
        [alice, bob],
        [],
    )

    from cognee.tests.utils.assert_graph_edges_present import assert_graph_edges_present
    from cognee.tests.utils.assert_graph_edges_not_present import assert_graph_edges_not_present
    from cognee.tests.utils.assert_edges_vector_index_present import (
        assert_edges_vector_index_present,
    )
    from cognee.tests.utils.assert_edges_vector_index_not_present import (
        assert_edges_vector_index_not_present,
    )

    edge = (alice.id, bob.id, "collaborates_with", {"edge_text": "collaborates with"})
    await assert_graph_edges_present([edge])
    await assert_edges_vector_index_present([edge])

    async with _dataset_context(dataset.id, dataset.owner_id):
        await cognee.forget(data_id=data_id_a, dataset_id=dataset.id, user=user)

    # Endpoint nodes survive (kept by data item B); the edge is hard-deleted.
    await assert_graph_nodes_present([alice, bob])
    await assert_graph_edges_not_present([edge])
    await assert_edges_vector_index_not_present([edge])


# ---------------------------------------------------------------------------
# rollback removing only run-introduced artifacts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rollback_removes_only_run_introduced_artifacts(graph_native_environment):
    """Rolling back one run removes only what that run solely introduced, leaving a
    prior run's artifacts (and shared nodes) intact in graph + vector stores."""
    user = await get_default_user()
    dataset = await create_authorized_dataset("gn_rollback_scope", user)

    add_old = await cognee.add("Baseline knowledge.", dataset_name=dataset.name, user=user)
    add_new = await cognee.add("New knowledge to roll back.", dataset_name=dataset.name, user=user)
    old_data_id = add_old.data_ingestion_info[0]["data_id"]
    new_data_id = add_new.data_ingestion_info[0]["data_id"]

    old_run = uuid4()
    new_run = uuid4()

    baseline = Person(name="Baseline")
    new_only = Person(name="NewOnly")
    shared = Person(name="Shared")  # touched by the old run, re-touched by the new run

    await _seed_shared_nodes(
        dataset,
        user,
        old_data_id,
        old_run,
        [baseline, shared],
        [(baseline.id, shared.id, "relates_to", {"edge_text": "relates to"})],
    )
    await _seed_shared_nodes(
        dataset,
        user,
        new_data_id,
        new_run,
        [new_only, shared],
        [(new_only.id, shared.id, "relates_to", {"edge_text": "relates to"})],
    )

    await assert_graph_nodes_present([baseline, new_only, shared])

    async with _dataset_context(dataset.id, dataset.owner_id):
        await cognify_rollback_handler(
            pipeline_run_id=new_run,
            dataset=dataset,
            user=user,
            data_ingestion_info=[{"data_id": str(new_data_id)}],
        )

    # Only the node the new run solely introduced is gone.
    await assert_graph_nodes_not_present([new_only])
    await assert_graph_nodes_present([baseline, shared])
    await assert_nodes_vector_index_not_present([new_only])
    await assert_nodes_vector_index_present([baseline, shared])

    # The shared node keeps the old run's source ref; rolling back the new run
    # must not strip the data-item ownership the old run established.
    old_ref = make_source_ref(dataset.id, old_data_id)
    new_ref = make_source_ref(dataset.id, new_data_id)
    shared_refs = await _node_source_refs(shared.id)
    assert old_ref in shared_refs
    assert new_ref not in shared_refs


# ---------------------------------------------------------------------------
# retry convergence after an injected vector failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vector_failure_leaves_graph_untouched_then_retry_converges(
    graph_native_environment, monkeypatch
):
    """A vector-delete failure during delete leaves graph provenance untouched
    (vectors are deleted first); retrying after the failure clears converges to
    the clean final state across graph + vector stores."""
    user = await get_default_user()
    dataset = await create_authorized_dataset("gn_retry_converge", user)

    add_a = await cognee.add("Solely owned data.", dataset_name=dataset.name, user=user)
    data_id_a = add_a.data_ingestion_info[0]["data_id"]

    solely = Person(name="Solely")
    await _seed_shared_nodes(dataset, user, data_id_a, uuid4(), [solely], [])

    await assert_graph_nodes_present([solely])
    await assert_nodes_vector_index_present([solely])

    ref_a = make_source_ref(dataset.id, data_id_a)

    # Inject a one-shot vector-delete failure on the engine actually in use.
    vector_engine = get_vector_engine()
    original_delete = vector_engine.delete_data_points
    state = {"fail": True}

    async def _failing_delete(collection_name, data_point_ids, *args, **kwargs):
        if state["fail"] and collection_name == "Person_name":
            raise RuntimeError("injected vector delete failure on Person_name")
        return await original_delete(collection_name, data_point_ids, *args, **kwargs)

    monkeypatch.setattr(vector_engine, "delete_data_points", _failing_delete)

    unified = await get_unified_engine()
    assert unified.supports_graph_native_delete()

    with pytest.raises(RuntimeError, match="injected vector delete failure"):
        await unified.delete_by_source_ref(ref_a)

    # Vectors are deleted first, so on failure the graph provenance is untouched:
    # the node and its ref are still present and the artifact is recoverable.
    await assert_graph_nodes_present([solely])
    assert ref_a in await _node_source_refs(solely.id)

    # Clear the failure and retry; the planner is retry-safe and converges.
    state["fail"] = False
    result = await unified.delete_by_source_ref(ref_a)

    assert result.nodes_deleted == 1
    await assert_graph_nodes_not_present([solely])
    await assert_nodes_vector_index_not_present([solely])
