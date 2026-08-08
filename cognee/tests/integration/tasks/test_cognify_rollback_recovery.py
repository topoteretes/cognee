import asyncio
import importlib
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.cognify.recovery import recover_stale_cognify_runs_on_startup
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.data.models import Data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.graph.models import Edge, Node
from cognee.modules.pipelines.models import PipelineContext, PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import create_user, get_default_user
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tests.utils.assert_graph_nodes_not_present import assert_graph_nodes_not_present
from cognee.tests.utils.assert_graph_nodes_present import assert_graph_nodes_present


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture
async def clean_test_environment(request, tmp_path, monkeypatch):
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
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

    add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

    async def _noop_index(*_args, **_kwargs):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)

    # This suite validates the relational-ledger rollback path (still the path
    # for every graph backend that lacks provenance support — Neo4j, Neptune,
    # Postgres). On the default Ladybug stack an empty graph would otherwise be
    # marked graph-provenance and skip the ledger entirely. Force the ledger
    # path so add_data_points writes the Node/Edge rows these tests assert on;
    # the rollback handler then also reads the (unmarked) graph as ledger-mode.
    # Graph-provenance rollback recovery is covered separately (test_rollback.py,
    # test_graph_provenance_delete_default_stack.py).
    async def _force_ledger(_graph_engine):
        return False

    monkeypatch.setattr(add_data_points_module, "mark_graph_provenance_if_empty", _force_ledger)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


async def _get_data_record(data_id):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return await session.get(Data, data_id)


async def _count_nodes_edges_for_run(dataset_id, pipeline_run_id):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        if pipeline_run_id is None:
            nodes = (
                (
                    await session.execute(
                        select(Node).where(
                            Node.dataset_id == dataset_id,
                            Node.pipeline_run_id.is_(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
            edges = (
                (
                    await session.execute(
                        select(Edge).where(
                            Edge.dataset_id == dataset_id,
                            Edge.pipeline_run_id.is_(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
        else:
            nodes = (
                (
                    await session.execute(
                        select(Node).where(
                            Node.dataset_id == dataset_id,
                            Node.pipeline_run_id == pipeline_run_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            edges = (
                (
                    await session.execute(
                        select(Edge).where(
                            Edge.dataset_id == dataset_id,
                            Edge.pipeline_run_id == pipeline_run_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
    return nodes, edges


@asynccontextmanager
async def _dataset_context(dataset_id, owner_id):
    async with set_database_global_context_variables(dataset_id, owner_id):
        yield


async def _mock_structured_output(
    _text_input: str,
    _system_prompt: str,
    response_model,
    **_kwargs,
):
    from cognee.shared.data_models import Edge as KGEdge
    from cognee.shared.data_models import KnowledgeGraph, Node as KGNode, SummarizedContent

    if response_model == SummarizedContent:
        return SummarizedContent(
            summary="Summary for rollback test",
            description="Summary for rollback test",
        )
    if response_model == KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[
                KGNode(id="Alice", name="Alice", type="Person", description="Person A"),
                KGNode(id="Bob", name="Bob", type="Person", description="Person B"),
            ],
            edges=[
                KGEdge(
                    source_node_id="Alice",
                    target_node_id="Bob",
                    relationship_name="knows",
                )
            ],
        )
    return "test"


@pytest.mark.asyncio
async def test_cognify_rollback_integration_keeps_preexisting_data_when_pipeline_fails_mid_execution(
    clean_test_environment, monkeypatch
):
    # Test 1: add->cognify (failure mid execution), rollback removes only failed run artifacts.
    user = await get_default_user()
    dataset_name = "rollback_integration_dataset"
    dataset = await create_authorized_dataset(dataset_name, user)

    add_old = await cognee.add("Older baseline data", dataset_name=dataset_name, user=user)
    old_data_id = add_old.data_ingestion_info[0]["data_id"]

    preexisting_run_id = uuid4()
    preexisting_nodes = [Person(name="Legacy-A"), Person(name="Legacy-B")]
    preexisting_edge = [
        (preexisting_nodes[0].id, preexisting_nodes[1].id, "related_to", {"edge_text": "legacy"})
    ]

    async with _dataset_context(dataset.id, dataset.owner_id):
        await add_data_points(
            preexisting_nodes,
            custom_edges=preexisting_edge,
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=old_data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=preexisting_run_id,
            ),
        )

    nodes_before, edges_before = await _count_nodes_edges_for_run(dataset.id, preexisting_run_id)
    assert len(nodes_before) >= 2
    assert len(edges_before) >= 1

    add_new = await cognee.add(
        "Data that will fail during cognify", dataset_name=dataset_name, user=user
    )
    assert add_new is not None

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    original_get_default_tasks = cognify_module.get_default_tasks

    async def _forced_failure_task(*_args, **_kwargs):
        raise RuntimeError("forced failure after default cognify tasks")

    async def _patched_get_default_tasks(*args, **kwargs):
        tasks = await original_get_default_tasks(*args, **kwargs)
        # Append a deterministic failure task to avoid coupling this test to any single built-in task.
        tasks.append(Task(_forced_failure_task))
        return tasks

    monkeypatch.setattr(cognify_module, "get_default_tasks", _patched_get_default_tasks)

    cognify_result = await cognee.cognify(datasets=[dataset_name], user=user)
    run_info = cognify_result[dataset.id]
    assert run_info.status == "PipelineRunErrored"

    failed_run_id = run_info.pipeline_run_id
    failed_nodes, failed_edges = await _count_nodes_edges_for_run(dataset.id, failed_run_id)
    assert failed_nodes == []
    assert failed_edges == []

    # Pre-existing data from older run must remain untouched.
    remaining_old_nodes, remaining_old_edges = await _count_nodes_edges_for_run(
        dataset.id, preexisting_run_id
    )
    assert len(remaining_old_nodes) >= 2
    assert len(remaining_old_edges) >= 1


@pytest.mark.asyncio
async def test_cognify_rollback_scope_isolated_across_users_and_datasets(clean_test_environment):
    # Test 2: rollback for one pipeline/dataset/user must not remove another one.
    user_a = await get_default_user()
    user_b = await create_user(email="rollback_user_b@test.com", password="password123")

    dataset_a = await create_authorized_dataset("rollback_scope_dataset_a", user_a)
    dataset_b = await create_authorized_dataset("rollback_scope_dataset_b", user_b)

    add_a = await cognee.add("A dataset content", dataset_name=dataset_a.name, user=user_a)
    add_b = await cognee.add("B dataset content", dataset_name=dataset_b.name, user=user_b)
    data_id_a = add_a.data_ingestion_info[0]["data_id"]
    data_id_b = add_b.data_ingestion_info[0]["data_id"]

    run_a = uuid4()
    run_b = uuid4()
    nodes_a = [Person(name="A-1"), Person(name="A-2")]
    nodes_b = [Person(name="B-1"), Person(name="B-2")]

    async def _seed_dataset(dataset, user, data_id, pipeline_run_id, nodes):
        async with _dataset_context(dataset.id, dataset.owner_id):
            await add_data_points(
                nodes,
                custom_edges=[(nodes[0].id, nodes[1].id, "knows", {"edge_text": "knows"})],
                ctx=PipelineContext(
                    user=user,
                    dataset=dataset,
                    data_item=SimpleNamespace(id=data_id),
                    pipeline_name="cognify_pipeline",
                    pipeline_run_id=pipeline_run_id,
                ),
            )

    await asyncio.gather(
        _seed_dataset(dataset_a, user_a, data_id_a, run_a, nodes_a),
        _seed_dataset(dataset_b, user_b, data_id_b, run_b, nodes_b),
    )

    async with _dataset_context(dataset_a.id, dataset_a.owner_id):
        await cognify_rollback_handler(
            pipeline_run_id=run_a,
            dataset=dataset_a,
            user=user_a,
            data_ingestion_info=[{"data_id": str(data_id_a)}],
        )

    await assert_graph_nodes_not_present(nodes_a)
    await assert_graph_nodes_present(nodes_b)

    nodes_a_rows, edges_a_rows = await _count_nodes_edges_for_run(dataset_a.id, run_a)
    nodes_b_rows, edges_b_rows = await _count_nodes_edges_for_run(dataset_b.id, run_b)
    assert nodes_a_rows == []
    assert edges_a_rows == []
    assert len(nodes_b_rows) >= 2
    assert len(edges_b_rows) >= 1


@pytest.mark.asyncio
async def test_cognify_startup_recovery_rolls_back_stale_started_runs(clean_test_environment):
    # Test 4: startup recovery should rollback stale STARTED cognify runs.
    user = await get_default_user()
    dataset = await create_authorized_dataset("rollback_recovery_dataset", user)

    add_result = await cognee.add("Recovery test data", dataset_name=dataset.name, user=user)
    data_id = add_result.data_ingestion_info[0]["data_id"]

    stale_run_id = uuid4()
    recovery_nodes = [Person(name="Recovery-1"), Person(name="Recovery-2")]

    async with _dataset_context(dataset.id, dataset.owner_id):
        await add_data_points(
            recovery_nodes,
            custom_edges=[
                (recovery_nodes[0].id, recovery_nodes[1].id, "links", {"edge_text": "links"})
            ],
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=stale_run_id,
            ),
        )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            PipelineRun(
                pipeline_run_id=stale_run_id,
                pipeline_name="cognify_pipeline",
                pipeline_id=uuid4(),
                status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                dataset_id=dataset.id,
                run_info={"data": [str(data_id)]},
                # Mark the run as old enough to be considered stale; a freshly
                # started run is treated as live and intentionally not recovered.
                created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
        )
        await session.commit()

    await assert_graph_nodes_present(recovery_nodes)
    await recover_stale_cognify_runs_on_startup()
    await assert_graph_nodes_not_present(recovery_nodes)

    nodes_after, edges_after = await _count_nodes_edges_for_run(dataset.id, stale_run_id)
    assert nodes_after == []
    assert edges_after == []


@pytest.mark.asyncio
async def test_cognify_rollback_is_idempotent(clean_test_environment):
    # Test 5: calling rollback twice should be safe no-op on second invocation.
    user = await get_default_user()
    dataset = await create_authorized_dataset("rollback_idempotent_dataset", user)
    add_result = await cognee.add("Idempotency test data", dataset_name=dataset.name, user=user)
    data_id = add_result.data_ingestion_info[0]["data_id"]

    run_id = uuid4()
    nodes = [Person(name="Idempotent-1"), Person(name="Idempotent-2")]

    async with _dataset_context(dataset.id, dataset.owner_id):
        await add_data_points(
            nodes,
            custom_edges=[(nodes[0].id, nodes[1].id, "connects", {"edge_text": "connects"})],
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=run_id,
            ),
        )

    async with _dataset_context(dataset.id, dataset.owner_id):
        await cognify_rollback_handler(
            pipeline_run_id=run_id,
            dataset=dataset,
            user=user,
            data_ingestion_info=[{"data_id": str(data_id)}],
        )
        await cognify_rollback_handler(
            pipeline_run_id=run_id,
            dataset=dataset,
            user=user,
            data_ingestion_info=[{"data_id": str(data_id)}],
        )

    await assert_graph_nodes_not_present(nodes)
    nodes_after, edges_after = await _count_nodes_edges_for_run(dataset.id, run_id)
    assert nodes_after == []
    assert edges_after == []


@pytest.mark.asyncio
async def test_cognify_rollback_preserves_legacy_rows_without_pipeline_run_id(
    clean_test_environment,
):
    # Test 6: rollback of a new run must not delete legacy rows with NULL pipeline_run_id.
    user = await get_default_user()
    dataset = await create_authorized_dataset("rollback_legacy_dataset", user)

    add_legacy = await cognee.add("Legacy data", dataset_name=dataset.name, user=user)
    add_new = await cognee.add("New data", dataset_name=dataset.name, user=user)
    legacy_data_id = add_legacy.data_ingestion_info[0]["data_id"]
    new_data_id = add_new.data_ingestion_info[0]["data_id"]

    legacy_nodes = [Person(name="Legacy-1"), Person(name="Legacy-2")]
    new_nodes = [Person(name="New-1"), Person(name="New-2")]
    new_run_id = uuid4()

    async with _dataset_context(dataset.id, dataset.owner_id):
        await add_data_points(
            legacy_nodes,
            custom_edges=[
                (legacy_nodes[0].id, legacy_nodes[1].id, "legacy", {"edge_text": "legacy"})
            ],
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=legacy_data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=None,
            ),
        )
        await add_data_points(
            new_nodes,
            custom_edges=[(new_nodes[0].id, new_nodes[1].id, "new", {"edge_text": "new"})],
            ctx=PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(id=new_data_id),
                pipeline_name="cognify_pipeline",
                pipeline_run_id=new_run_id,
            ),
        )

    legacy_rel_nodes_before, legacy_rel_edges_before = await _count_nodes_edges_for_run(
        dataset.id, None
    )
    new_rel_nodes_before, new_rel_edges_before = await _count_nodes_edges_for_run(
        dataset.id, new_run_id
    )
    assert len(legacy_rel_nodes_before) >= 2
    assert len(legacy_rel_edges_before) >= 1
    assert len(new_rel_nodes_before) >= 2
    assert len(new_rel_edges_before) >= 1

    async with _dataset_context(dataset.id, dataset.owner_id):
        await cognify_rollback_handler(
            pipeline_run_id=new_run_id,
            dataset=dataset,
            user=user,
            data_ingestion_info=[{"data_id": str(new_data_id)}],
        )

    await assert_graph_nodes_present(legacy_nodes)
    await assert_graph_nodes_not_present(new_nodes)

    legacy_rel_nodes_after, legacy_rel_edges_after = await _count_nodes_edges_for_run(
        dataset.id, None
    )
    new_rel_nodes_after, new_rel_edges_after = await _count_nodes_edges_for_run(
        dataset.id, new_run_id
    )
    assert len(legacy_rel_nodes_after) >= 2
    assert len(legacy_rel_edges_after) >= 1
    assert new_rel_nodes_after == []
    assert new_rel_edges_after == []

    # Legacy data should remain discoverable in relational metadata.
    legacy_data_record = await _get_data_record(legacy_data_id)
    assert legacy_data_record is not None
