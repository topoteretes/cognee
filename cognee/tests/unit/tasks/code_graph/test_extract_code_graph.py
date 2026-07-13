import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import cognee
import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.tasks.code_graph.enola import parse_enola_snapshot
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_edges,
    build_code_graph_edges,
    extract_code_graph,
    fact_node_id,
    get_code_graph_tasks,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import (
    ApiEndpoint,
    CodeModule,
    CodeRepository,
    CodeService,
    CodeSymbol,
    ExternalDependency,
    StorageResource,
)

# importlib.import_module returns the real submodule even though the package
# __init__ re-exports same-named functions that shadow the submodule attribute.
extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture_facts():
    facts, _receipt = parse_enola_snapshot(FIXTURES_DIR)
    return facts


def test_every_kind_maps_to_the_right_data_point_class():
    data_points = map_facts_to_data_points(_fixture_facts(), repo_path="/tmp/shop")

    by_name = {data_point.name: data_point for data_point in data_points}

    assert isinstance(by_name["acme/shop"], CodeRepository)
    assert isinstance(by_name["app"], CodeModule)
    assert isinstance(by_name["main"], CodeSymbol)
    assert isinstance(by_name["GET /orders"], ApiEndpoint)
    assert isinstance(by_name["orders_table"], StorageResource)
    assert isinstance(by_name["github.com/lib/pq"], ExternalDependency)
    assert isinstance(by_name["shop-api"], CodeService)

    # 16 facts + the CodeRepository node.
    assert len(data_points) == 17


def test_symbol_kind_and_common_fields_are_mapped():
    data_points = map_facts_to_data_points(_fixture_facts(), repo_path="/tmp/shop")
    by_name = {data_point.name: data_point for data_point in data_points}

    main = by_name["main"]
    assert main.symbol_kind == "function"
    assert main.file_path == "app/main.go"
    assert main.line == 12
    assert main.repo == "acme/shop"
    assert main.part_of is by_name["acme/shop"]

    symbol_kinds = {
        data_point.symbol_kind for data_point in data_points if isinstance(data_point, CodeSymbol)
    }
    assert symbol_kinds == {
        "function",
        "method",
        "struct",
        "interface",
        "class",
        "constant",
        "variable",
        "enum",
        "type",
    }


@pytest.mark.asyncio
async def test_extract_code_graph_ids_are_deterministic_across_runs():
    first_run = await extract_code_graph(snapshot_dir=FIXTURES_DIR, repo_path="/tmp/shop")
    second_run = await extract_code_graph(snapshot_dir=FIXTURES_DIR, repo_path="/tmp/shop")

    first_ids = sorted(str(data_point.id) for data_point in first_run)
    second_ids = sorted(str(data_point.id) for data_point in second_run)

    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_run)


def test_build_code_graph_edges_resolves_and_skips():
    edges, skipped = build_code_graph_edges(_fixture_facts())

    # One relation targets nonexistent "ghost_function", one has
    # un-normalizable keys ({"foo": "bar"}).
    assert skipped == 2

    relationship_names = {edge[2] for edge in edges}
    assert relationship_names == {
        "declares",
        "imports",
        "calls",
        "instantiates",
        "has_method",
        "implements",
        "handled_by",
        "depends_on",
        "injects",
    }
    assert len(edges) == 9
    assert not any("ghost_function" in edge[1] for edge in edges)


def test_edge_tuples_have_correct_source_relationship_target():
    edges, _skipped = build_code_graph_edges(_fixture_facts())

    expected_source = str(fact_node_id("acme/shop", "symbol", "main"))
    expected_target = str(fact_node_id("acme/shop", "symbol", "helper"))

    calls_edges = [edge for edge in edges if edge[2] == "calls"]
    assert len(calls_edges) == 1

    source_id, target_id, relationship_name, properties = calls_edges[0]
    assert source_id == expected_source
    assert target_id == expected_target
    assert relationship_name == "calls"
    assert properties["source_node_id"] == expected_source
    assert properties["target_node_id"] == expected_target
    assert properties["relationship_name"] == "calls"


def test_unknown_kind_and_missing_name_facts_are_skipped():
    facts = [
        {"kind": "unknown_kind", "name": "x"},
        {"kind": "symbol"},
        {"kind": "symbol", "name": ""},
        {"kind": "module", "name": "app", "repo": "acme/shop"},
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/tmp/shop")

    names = {data_point.name for data_point in data_points}
    # Only the repository node and the valid module survive.
    assert names == {"acme/shop", "app"}


def test_malformed_field_types_are_skipped_not_fatal():
    facts = [
        {"kind": ["module"], "name": "x"},
        {"kind": "module", "name": 42},
        {"kind": "module", "name": "m", "line": "abc", "repo": "acme/shop"},
        {"kind": "module", "name": "app", "file": 7, "props": ["not", "a", "dict"]},
        {"kind": "module", "name": "ok", "line": 3, "repo": "acme/shop"},
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/tmp/shop")
    by_name = {data_point.name: data_point for data_point in data_points}

    assert "ok" in by_name
    assert by_name["ok"].line == 3
    # Wrong-typed line and file are dropped, not fatal.
    assert by_name["m"].line is None
    assert by_name["app"].file_path is None
    assert "x" not in by_name
    assert 42 not in by_name

    # build_code_graph_edges tolerates the same malformed facts.
    edges, _skipped = build_code_graph_edges(facts, repo_path="/tmp/shop")
    assert edges == []


def test_node_and_edge_ids_agree_when_facts_have_no_repo_field():
    facts = [
        {"kind": "symbol", "name": "main", "relations": [{"type": "calls", "target": "helper"}]},
        {"kind": "symbol", "name": "helper"},
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/path/to/myrepo")
    edges, skipped = build_code_graph_edges(facts, repo_path="/path/to/myrepo")

    node_ids = {str(data_point.id) for data_point in data_points}
    assert skipped == 0
    assert len(edges) == 1
    source_id, target_id, _relationship_name, _properties = edges[0]
    assert source_id in node_ids
    assert target_id in node_ids
    assert source_id == str(fact_node_id("myrepo", "symbol", "main"))


def test_duplicate_names_in_same_repo_resolve_to_the_single_shared_node():
    facts = [
        {"kind": "symbol", "name": "init", "file": "a.go", "repo": "r"},
        {"kind": "symbol", "name": "init", "file": "b.go", "repo": "r"},
        {
            "kind": "symbol",
            "name": "main",
            "repo": "r",
            "relations": [{"type": "calls", "target": "init"}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    assert skipped == 0
    assert len(edges) == 1
    assert edges[0][1] == str(fact_node_id("r", "symbol", "init"))


def test_ambiguous_target_prefers_same_repo_else_skips():
    facts = [
        {"kind": "symbol", "name": "helper", "repo": "a"},
        {"kind": "symbol", "name": "helper", "repo": "b"},
        {
            "kind": "symbol",
            "name": "main",
            "repo": "a",
            "relations": [{"type": "calls", "target": "helper"}],
        },
        {
            "kind": "symbol",
            "name": "other",
            "repo": "c",
            "relations": [{"type": "calls", "target": "helper"}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    # The repo-a relation resolves to repo-a's helper; the repo-c one stays ambiguous.
    assert skipped == 1
    assert len(edges) == 1
    assert edges[0][0] == str(fact_node_id("a", "symbol", "main"))
    assert edges[0][1] == str(fact_node_id("a", "symbol", "helper"))


def test_duplicated_relation_emits_a_single_edge():
    facts = [
        {
            "kind": "symbol",
            "name": "main",
            "repo": "r",
            "relations": [
                {"type": "calls", "target": "helper"},
                {"type": "calls", "target": "helper"},
            ],
        },
        {"kind": "symbol", "name": "helper", "repo": "r"},
    ]

    edges, skipped = build_code_graph_edges(facts)

    assert skipped == 0
    assert len(edges) == 1


def test_multi_repo_snapshot_creates_one_repository_per_repo():
    facts = [
        {"kind": "module", "name": "checkout", "repo": "acme/shop"},
        {"kind": "module", "name": "invoices", "repo": "acme/billing"},
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/tmp/shop")
    by_name = {data_point.name: data_point for data_point in data_points}

    assert isinstance(by_name["acme/shop"], CodeRepository)
    assert isinstance(by_name["acme/billing"], CodeRepository)
    assert by_name["checkout"].part_of is by_name["acme/shop"]
    assert by_name["invoices"].part_of is by_name["acme/billing"]
    assert by_name["acme/billing"].id == fact_node_id("acme/billing", "repository", "acme/billing")


@pytest.mark.asyncio
async def test_extract_code_graph_accepts_repo_path_as_positional_payload(monkeypatch):
    received = {}

    async def fake_run_enola_generate(repo_path, timeout=600.0):
        received["repo_path"] = repo_path
        return FIXTURES_DIR

    monkeypatch.setattr(extract_module, "run_enola_generate", fake_run_enola_generate)

    data_points = await extract_code_graph("/tmp/shop")

    assert received["repo_path"] == "/tmp/shop"
    assert len(data_points) == 17


@pytest.mark.asyncio
async def test_extract_code_graph_without_repo_path_or_snapshot_dir_raises():
    with pytest.raises(ValueError):
        await extract_code_graph()


@pytest.mark.asyncio
async def test_add_code_graph_edges_writes_edges_and_passes_data_points_through(monkeypatch):
    graph_engine_module = importlib.import_module(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )

    graph_engine = AsyncMock()
    monkeypatch.setattr(
        graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph_engine)
    )

    data_points = ["sentinel"]
    result = await add_code_graph_edges(data_points, snapshot_dir=FIXTURES_DIR)

    assert result is data_points
    graph_engine.add_edges.assert_awaited_once()
    (edges,) = graph_engine.add_edges.await_args.args
    expected_edges, _skipped = build_code_graph_edges(_fixture_facts())
    assert len(edges) == 9
    assert [edge[:3] for edge in edges] == [edge[:3] for edge in expected_edges]


@pytest.mark.asyncio
async def test_add_code_graph_edges_registers_edges_in_rollback_ledger(monkeypatch):
    graph_engine_module = importlib.import_module(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )
    graph_methods_module = importlib.import_module("cognee.modules.graph.methods")

    graph_engine = AsyncMock()
    monkeypatch.setattr(
        graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph_engine)
    )
    upsert_edges_mock = AsyncMock()
    monkeypatch.setattr(graph_methods_module, "upsert_edges", upsert_edges_mock)

    ctx = SimpleNamespace(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )

    await add_code_graph_edges([], snapshot_dir=FIXTURES_DIR, ctx=ctx)

    upsert_edges_mock.assert_awaited_once()
    call = upsert_edges_mock.await_args
    assert len(call.args[0]) == 9
    assert call.kwargs["tenant_id"] == ctx.user.tenant_id
    assert call.kwargs["user_id"] == ctx.user.id
    assert call.kwargs["dataset_id"] == ctx.dataset.id
    assert call.kwargs["data_id"] == ctx.data_item.id
    assert call.kwargs["pipeline_run_id"] == ctx.pipeline_run_id


@pytest.mark.asyncio
async def test_public_code_graph_pipeline_accepts_repo_path_payload_with_access_control(
    monkeypatch,
):
    custom_pipeline_module = importlib.import_module(
        "cognee.modules.run_custom_pipeline.run_custom_pipeline"
    )
    add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    graph_engine_module = importlib.import_module(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )
    graph_methods_module = importlib.import_module("cognee.modules.graph.methods")

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")

    graph_engine = AsyncMock()
    vector_engine = MagicMock()
    unified_engine = SimpleNamespace(
        graph=graph_engine,
        vector=vector_engine,
        has_capability=MagicMock(return_value=False),
    )

    async def fake_get_graph_from_model(data_point, **_kwargs):
        return [data_point], []

    monkeypatch.setattr(
        add_data_points_module,
        "get_graph_from_model",
        AsyncMock(side_effect=fake_get_graph_from_model),
    )
    monkeypatch.setattr(
        add_data_points_module,
        "deduplicate_nodes_and_edges",
        lambda nodes, edges: (nodes, edges),
    )
    monkeypatch.setattr(
        add_data_points_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(add_data_points_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", AsyncMock())

    storage_upsert_nodes = AsyncMock()
    storage_upsert_edges = AsyncMock()
    monkeypatch.setattr(add_data_points_module, "upsert_nodes", storage_upsert_nodes)
    monkeypatch.setattr(add_data_points_module, "upsert_edges", storage_upsert_edges)
    monkeypatch.setattr(
        graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph_engine)
    )
    code_graph_upsert_edges = AsyncMock()
    monkeypatch.setattr(graph_methods_module, "upsert_edges", code_graph_upsert_edges)

    async def execute_in_process(**kwargs):
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4(), email="test@example.com")
        ctx = PipelineContext(
            user=user,
            dataset=SimpleNamespace(id=uuid4(), name=kwargs["datasets"]),
            data_item=kwargs["data"],
            pipeline_run_id=uuid4(),
            pipeline_name=kwargs["pipeline_name"],
        )
        results = []
        async for result in run_tasks_base(kwargs["tasks"], [kwargs["data"]], user, ctx):
            results.append(result)
        return results

    monkeypatch.setattr(
        custom_pipeline_module,
        "get_pipeline_executor",
        lambda run_in_background: execute_in_process,
    )

    repo_path = "/tmp/shop"
    result = await cognee.run_custom_pipeline(
        tasks=get_code_graph_tasks(repo_path, snapshot_dir=FIXTURES_DIR),
        data=repo_path,
        dataset="code_graph_demo",
        pipeline_name="code_graph_pipeline",
    )

    assert len(result) == 1
    assert len(result[0]) == 17
    graph_engine.add_nodes.assert_awaited_once()
    assert graph_engine.add_edges.await_count == 2
    storage_upsert_nodes.assert_not_awaited()
    storage_upsert_edges.assert_not_awaited()
    code_graph_upsert_edges.assert_not_awaited()
