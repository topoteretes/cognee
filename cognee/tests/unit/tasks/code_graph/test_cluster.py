"""A dataset's code repositories are extracted as one enola cluster."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.code_graph.cluster import (
    ClusterMember,
    DatasetCodeSnapshot,
    code_repo_path,
    load_dataset_cluster,
    refresh_dataset_code_graph,
    refresh_dataset_code_graph_after_delete,
    resolve_member_labels,
    snapshot_dataset_code_repos,
    write_cluster_config,
)
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    extract_code_graph,
    fact_node_id,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeRepository

cluster_module = importlib.import_module("cognee.tasks.code_graph.cluster")
extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
data_methods_module = importlib.import_module("cognee.modules.data.methods")
pipeline_module = importlib.import_module("cognee.modules.run_custom_pipeline")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

SNAPSHOT_ID = "sha256:c1u5te4"

# A frontend function calls a backend route; the backend route has a handler.
CLUSTER_FACTS = [
    {"kind": "symbol", "name": "api.fetchRounds", "repo": "ui", "id": "a" * 32},
    {
        "kind": "route",
        "name": "/api/rounds",
        "repo": "ui",
        "props": {
            "role": "client",
            "caller": "api.fetchRounds",
            "caller_id": "a" * 32,
            "matched_routes": [
                {"repo": "golf", "name": "/api/rounds", "confidence": "verified", "id": "b" * 32}
            ],
        },
    },
    {
        "kind": "route",
        "name": "/api/rounds",
        "repo": "golf",
        "id": "b" * 32,
        "relations": [{"kind": "handled_by", "target": "handlers.ListRounds"}],
    },
    {"kind": "symbol", "name": "handlers.ListRounds", "repo": "golf"},
]

UI_ROUTE = str(fact_node_id("ui", "route", "/api/rounds"))
GOLF_ROUTE = str(fact_node_id("golf", "route", "/api/rounds"))
REACHES = (UI_ROUTE, GOLF_ROUTE, "reaches_route")


def _write_snapshot(directory, facts=CLUSTER_FACTS, snapshot_id=SNAPSHOT_ID, remote=None):
    snapshot_dir = directory / ".enola"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    receipt = {"snapshot_id": snapshot_id}
    if remote:
        receipt["git"] = {"remote": remote}
    (snapshot_dir / "receipt.json").write_text(json.dumps(receipt))
    return snapshot_dir


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_graph_data.return_value = (list(nodes), list(edges))
    engine.get_node.return_value = None
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(extract_module, "_invalidate_code_graph_snapshot", lambda ctx=None: None)
    return engine


def _row(repo_path):
    return SimpleNamespace(
        id=uuid4(), system_metadata={"source": "code_repo", "repo_path": str(repo_path)}
    )


def _repos(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.mkdir()
        paths.append(path)
    return paths


# --- the cluster config and its members ---------------------------------------


def test_cluster_config_lists_absolute_paths_under_the_dataset_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_module, "cluster_directory", lambda dataset_id: tmp_path / "c")
    odd = tmp_path / "re: po"

    config_path = write_cluster_config(uuid4(), [tmp_path / "golf", odd])

    assert config_path == tmp_path / "c" / "cluster.yaml"
    assert config_path.read_text() == (
        f'repos:\n  - "{tmp_path / "golf"}"\n  - {json.dumps(str(odd))}\n'
    )


def test_cluster_directory_falls_back_to_the_clones_directory_for_remote_storage(monkeypatch):
    base_config_module = importlib.import_module("cognee.base_config")
    config = SimpleNamespace(system_root_directory="s3://bucket/system", repos_root_directory="/r")
    monkeypatch.setattr(base_config_module, "get_base_config", lambda: config)
    dataset_id = uuid4()

    assert str(cluster_module.cluster_directory(dataset_id)) == f"/r/code_clusters/{dataset_id}"


def test_code_repo_path_reads_metadata_as_dict_or_json_string(tmp_path):
    metadata = {"source": "code_repo", "repo_path": str(tmp_path)}

    assert code_repo_path(SimpleNamespace(system_metadata=metadata)) == tmp_path
    assert code_repo_path(SimpleNamespace(system_metadata=json.dumps(metadata))) == tmp_path
    assert code_repo_path(SimpleNamespace(system_metadata=None)) is None


def test_member_label_is_the_remote_name_else_the_directory_name(tmp_path):
    golf, ui = _repos(tmp_path, "checkout-1", "ui")
    _write_snapshot(golf, remote="github.com/acme/golf")
    _write_snapshot(ui)

    assert resolve_member_labels([golf, ui], {"golf", "ui"}) == ["golf", "ui"]


def test_member_label_follows_the_facts_when_enola_fell_back_to_the_directory(tmp_path):
    """Two remotes named alike: enola labels the second one by its directory."""
    first, second = _repos(tmp_path, "web-a", "web-b")
    _write_snapshot(first, remote="github.com/acme/web")
    _write_snapshot(second, remote="github.com/other/web")

    assert resolve_member_labels([first, second], {"web", "web-b"}) == ["web", "web-b"]


def test_members_that_cannot_be_told_apart_are_not_a_cluster(tmp_path):
    """Two checkouts sharing both a remote name and a directory name."""
    first = tmp_path / "one" / "web"
    second = tmp_path / "two" / "web"
    _write_snapshot(first, remote="github.com/acme/web")
    _write_snapshot(second, remote="github.com/other/web")

    assert resolve_member_labels([first, second], {"web"}) is None


@pytest.mark.asyncio
async def test_a_single_repository_is_not_snapshotted_as_a_cluster(monkeypatch, tmp_path):
    (golf,) = _repos(tmp_path, "golf")
    run_cluster = AsyncMock()
    monkeypatch.setattr(cluster_module, "run_enola_cluster", run_cluster)
    repos = [(_row(golf), golf), (_row(tmp_path / "gone"), tmp_path / "gone")]

    assert await snapshot_dataset_code_repos(uuid4(), repos) is None
    run_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_snapshot_runs_enola_once_and_labels_each_member(monkeypatch, tmp_path):
    golf, ui = _repos(tmp_path, "golf", "ui")
    snapshot_dir = _write_snapshot(golf)
    _write_snapshot(ui)
    monkeypatch.setattr(cluster_module, "cluster_directory", lambda dataset_id: tmp_path / "c")
    run_cluster = AsyncMock(return_value=snapshot_dir)
    monkeypatch.setattr(cluster_module, "run_enola_cluster", run_cluster)
    golf_row, ui_row = _row(golf), _row(ui)

    snapshot = await snapshot_dataset_code_repos(
        uuid4(), [(golf_row, golf), (ui_row, ui)], timeout=10.0
    )

    run_cluster.assert_awaited_once_with(tmp_path / "c" / "cluster.yaml", [golf, ui], timeout=20.0)
    assert snapshot.snapshot_dir == snapshot_dir
    assert [(member.data, member.label) for member in snapshot.members] == [
        (golf_row, "golf"),
        (ui_row, "ui"),
    ]


# --- loading one member of a cluster snapshot ---------------------------------


def test_scoped_mapping_keeps_one_repository_with_its_real_path(tmp_path):
    data_points = map_facts_to_data_points(CLUSTER_FACTS, repo_path=tmp_path, repo_scope="ui")

    repositories = [point for point in data_points if isinstance(point, CodeRepository)]
    assert [(repo.name, repo.path) for repo in repositories] == [("ui", str(tmp_path))]
    assert {point.repo for point in data_points if not isinstance(point, CodeRepository)} == {"ui"}


@pytest.mark.asyncio
async def test_member_skip_check_reads_its_own_repository_node(monkeypatch, tmp_path):
    snapshot_dir = _write_snapshot(tmp_path)
    engine = _mock_engine(monkeypatch)
    engine.get_node.return_value = {"last_snapshot_id": SNAPSHOT_ID}

    assert await extract_code_graph(snapshot_dir=snapshot_dir, repo_scope="ui") == []
    engine.get_node.assert_awaited_once_with(str(fact_node_id("ui", "repository", "ui")))


async def _load_member(snapshot_dir, repo_scope, tmp_path):
    data_points = await extract_code_graph(
        repo_path=tmp_path, snapshot_dir=snapshot_dir, repo_scope=repo_scope
    )
    state = await add_code_graph_data_points(data_points)
    await add_code_graph_edges(
        state, repo_path=tmp_path, snapshot_dir=snapshot_dir, repo_scope=repo_scope
    )


def _written_edges(engine):
    return {
        (str(source), str(target), relationship)
        for call in engine.add_edges.await_args_list
        for source, target, relationship, _properties in call.args[0]
    }


@pytest.mark.asyncio
async def test_edge_into_a_member_not_loaded_yet_waits_for_that_member(monkeypatch, tmp_path):
    snapshot_dir = _write_snapshot(tmp_path)
    monkeypatch.setattr(extract_module, "add_data_points", AsyncMock(), raising=False)
    storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    monkeypatch.setattr(storage_module, "add_data_points", AsyncMock())

    # The frontend loads first: the backend route it reaches is not in the graph.
    engine = _mock_engine(monkeypatch)
    await _load_member(snapshot_dir, "ui", tmp_path)

    written = _written_edges(engine)
    assert REACHES not in written
    assert {relationship for _source, _target, relationship in written} == {"makes_request"}
    # Only the member that loaded is stamped; a stamp on the other would skip its load.
    stamped = [repo.name for call in engine.add_nodes.await_args_list for repo in call.args[0]]
    assert stamped == ["ui"]

    # The backend loads next, with the frontend's nodes in the graph: it writes
    # the edge between them along with its own.
    ui_nodes = [
        (str(fact_node_id("ui", kind, name)), {"type": "CodeSymbol", "repo": "ui"})
        for kind, name in (("symbol", "api.fetchRounds"), ("route", "/api/rounds"))
    ]
    engine = _mock_engine(monkeypatch, nodes=ui_nodes)
    await _load_member(snapshot_dir, "golf", tmp_path)

    written = _written_edges(engine)
    assert REACHES in written
    assert {relationship for _source, _target, relationship in written} == {
        "reaches_route",
        "handled_by",
    }


@pytest.mark.asyncio
async def test_member_load_does_not_sweep_the_other_members_edges(monkeypatch, tmp_path):
    snapshot_dir = _write_snapshot(tmp_path)
    storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    monkeypatch.setattr(storage_module, "add_data_points", AsyncMock())
    golf_nodes = [
        (str(fact_node_id("golf", kind, name)), {"type": "CodeSymbol", "repo": "golf"})
        for kind, name in (("route", "/api/rounds"), ("symbol", "handlers.ListRounds"))
    ]
    handled_by = (
        GOLF_ROUTE,
        str(fact_node_id("golf", "symbol", "handlers.ListRounds")),
        "handled_by",
        {},
    )
    engine = _mock_engine(monkeypatch, nodes=golf_nodes, edges=[handled_by])

    await _load_member(snapshot_dir, "ui", tmp_path)

    engine.delete_edge_triples.assert_not_awaited()
    engine.delete_nodes.assert_not_awaited()
    assert REACHES in _written_edges(engine)


# --- rebuilding a dataset -----------------------------------------------------


@pytest.fixture
def refresh_env(monkeypatch, tmp_path):
    golf, ui = _repos(tmp_path, "golf", "ui")
    golf_row, ui_row = _row(golf), _row(ui)
    snapshot = DatasetCodeSnapshot(
        snapshot_dir=golf / ".enola",
        members=[
            ClusterMember(data=golf_row, repo_path=golf, label="golf"),
            ClusterMember(data=ui_row, repo_path=ui, label="ui"),
        ],
    )
    snapshot_mock = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(cluster_module, "snapshot_dataset_code_repos", snapshot_mock)
    rows_mock = AsyncMock(return_value=[golf_row, ui_row])
    monkeypatch.setattr(cluster_module, "dataset_code_repo_rows", rows_mock)
    pipeline_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(pipeline_module, "run_custom_pipeline", pipeline_mock)
    mark_mock = AsyncMock()
    monkeypatch.setattr(data_methods_module, "mark_data_processed", mark_mock)
    return SimpleNamespace(
        dataset=SimpleNamespace(id=uuid4()),
        user=SimpleNamespace(id=uuid4()),
        golf=golf,
        ui=ui,
        golf_row=golf_row,
        ui_row=ui_row,
        snapshot=snapshot,
        snapshot_mock=snapshot_mock,
        pipeline=pipeline_mock,
        mark=mark_mock,
    )


def _task_kwargs(call, index=0):
    return call.kwargs["tasks"][index].default_params["kwargs"]


@pytest.mark.asyncio
async def test_refresh_runs_each_member_over_its_own_row_from_one_snapshot(refresh_env):
    env = refresh_env

    runs = await refresh_dataset_code_graph(
        env.dataset, env.user, known_repos=[(env.ui_row, env.ui)]
    )

    # The row the caller just stored comes first; the dataset's other row joins it.
    env.snapshot_mock.assert_awaited_once()
    assert env.snapshot_mock.await_args.args[1] == [(env.ui_row, env.ui), (env.golf_row, env.golf)]
    assert [(run.label, run.error) for run in runs] == [("golf", None), ("ui", None)]
    calls = env.pipeline.await_args_list
    assert [call.kwargs["data"] for call in calls] == [[env.golf_row], [env.ui_row]]
    assert [_task_kwargs(call)["repo_scope"] for call in calls] == ["golf", "ui"]
    assert {str(_task_kwargs(call)["snapshot_dir"]) for call in calls} == {
        str(env.snapshot.snapshot_dir)
    }
    assert [_task_kwargs(call, 2)["repo_scope"] for call in calls] == ["golf", "ui"]
    assert [call.args[0] for call in env.mark.await_args_list] == [env.golf_row.id, env.ui_row.id]


@pytest.mark.asyncio
async def test_refresh_keeps_going_when_one_member_fails(refresh_env):
    env = refresh_env
    env.pipeline.side_effect = [RuntimeError("enola crashed"), None]

    runs = await refresh_dataset_code_graph(env.dataset, env.user)

    assert [str(run.error) if run.error else None for run in runs] == ["enola crashed", None]
    env.mark.assert_awaited_once()
    assert env.mark.await_args.args[0] == env.ui_row.id


@pytest.mark.asyncio
async def test_refresh_without_a_cluster_runs_each_repository_unscoped(refresh_env):
    env = refresh_env
    env.snapshot_mock.return_value = None

    runs = await refresh_dataset_code_graph(env.dataset, env.user)

    assert [run.label for run in runs] == [None, None]
    for call in env.pipeline.await_args_list:
        assert _task_kwargs(call)["repo_scope"] is None
        assert _task_kwargs(call)["snapshot_dir"] is None


@pytest.mark.asyncio
async def test_refresh_after_delete_never_raises(monkeypatch, refresh_env):
    env = refresh_env
    invalidated = []
    monkeypatch.setattr(
        code_retriever_module,
        "invalidate_code_graph_snapshot_cache",
        lambda dataset_id=None: invalidated.append(dataset_id),
    )
    env.snapshot_mock.side_effect = RuntimeError("enola missing")

    await refresh_dataset_code_graph_after_delete(env.dataset, env.user)

    env.pipeline.assert_not_awaited()
    assert invalidated == [env.dataset.id]


# --- the cognify CODE_REPO route ----------------------------------------------


@pytest.mark.asyncio
async def test_route_loads_every_member_under_its_own_row_once_per_run(monkeypatch, refresh_env):
    env = refresh_env
    cluster_module._run_snapshots.clear()
    extract = AsyncMock(return_value=["point"])
    load = AsyncMock(side_effect=lambda points, ctx=None, graph_only=True: points)
    edges = AsyncMock()
    monkeypatch.setattr(extract_module, "extract_code_graph", extract)
    monkeypatch.setattr(extract_module, "add_code_graph_data_points", load)
    monkeypatch.setattr(extract_module, "add_code_graph_edges", edges)
    ctx = PipelineContext(
        user=env.user, data_item=env.ui_row, dataset=env.dataset, pipeline_run_id=uuid4()
    )

    assert await load_dataset_cluster(ctx) is True
    # The dataset's second row reaches the route in the same pipeline run.
    assert await load_dataset_cluster(ctx) is True

    env.snapshot_mock.assert_awaited_once()
    assert [call.kwargs["repo_scope"] for call in extract.await_args_list[:2]] == ["golf", "ui"]
    assert [call.kwargs["ctx"].data_item for call in load.await_args_list[:2]] == [
        env.golf_row,
        env.ui_row,
    ]
    assert [call.kwargs["ctx"].data_item for call in edges.await_args_list[:2]] == [
        env.golf_row,
        env.ui_row,
    ]


@pytest.mark.asyncio
async def test_route_leaves_a_single_repository_to_the_caller(refresh_env):
    env = refresh_env
    cluster_module._run_snapshots.clear()
    env.snapshot_mock.return_value = None
    ctx = PipelineContext(user=env.user, data_item=env.ui_row, dataset=env.dataset)

    assert await load_dataset_cluster(ctx) is False
    assert await load_dataset_cluster(None) is False
