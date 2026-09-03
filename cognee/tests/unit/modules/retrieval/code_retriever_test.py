import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from cognee.modules.retrieval.code_retriever import (
    CODE_NODE_TYPES,
    CodeRetriever,
    CodeSearchValidationError,
    _CodeGraphSnapshot,
    _CodeGraphSnapshotCache,
    _code_graph_snapshot_cache_key,
    invalidate_code_graph_snapshot_cache,
)
from cognee.context_global_variables import current_dataset_id


WIDGET_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture(autouse=True)
def clear_code_graph_snapshot_cache():
    invalidate_code_graph_snapshot_cache(all_entries=True)
    yield
    invalidate_code_graph_snapshot_cache(all_entries=True)


def _code_graph():
    nodes = [
        (
            "repo-a",
            {"name": "repo-a", "type": "CodeRepository", "path": "/src/repo-a"},
        ),
        (
            WIDGET_ID,
            {
                "name": "pkg.Widget",
                "type": "CodeSymbol",
                "properties": json.dumps(
                    {
                        "file_path": "pkg/widget.py",
                        "line": 10,
                        "repo": "repo-a",
                        "symbol_kind": "class",
                        "fact_properties": {"exported": True, "language": "python"},
                    }
                ),
            },
        ),
        (
            "method",
            {
                "name": "pkg.Widget.run",
                "type": "CodeSymbol",
                "file_path": "pkg/widget.py",
                "line": 20,
                "repo": "repo-a",
                "symbol_kind": "method",
                "fact_properties": {"exported": False},
            },
        ),
        (
            "constructor",
            {
                "name": "pkg.NewWidget",
                "type": "CodeSymbol",
                "file_path": "pkg/widget.py",
                "line": 5,
                "repo": "repo-a",
                "symbol_kind": "function",
            },
        ),
        (
            "entry",
            {
                "name": "pkg.entry",
                "type": "CodeSymbol",
                "file_path": "pkg/main.py",
                "line": 7,
                "repo": "repo-a",
                "symbol_kind": "function",
            },
        ),
        (
            "handler",
            {
                "name": "api.handler",
                "type": "CodeSymbol",
                "file_path": "api/handler.py",
                "line": 9,
                "repo": "repo-b",
                "symbol_kind": "function",
            },
        ),
        (
            "route",
            {
                "name": "GET /widgets",
                "type": "ApiEndpoint",
                "file_path": "api/routes.py",
                "line": 3,
                "repo": "repo-b",
                "fact_properties": {"method": "GET"},
            },
        ),
        (
            "database",
            {
                "name": "widget_db",
                "type": "StorageResource",
                "file_path": "pkg/storage.py",
                "repo": "repo-a",
            },
        ),
        (
            "helper-a",
            {
                "name": "shared.Helper",
                "type": "CodeSymbol",
                "file_path": "a/helper.py",
                "repo": "repo-a",
                "symbol_kind": "class",
            },
        ),
        (
            "helper-b",
            {
                "name": "shared.Helper",
                "type": "CodeSymbol",
                "file_path": "b/helper.py",
                "repo": "repo-b",
                "symbol_kind": "class",
            },
        ),
        (
            "noise",
            {"name": "ordinary text", "type": "DocumentChunk", "text": "not code"},
        ),
    ]
    edges = [
        (WIDGET_ID, "method", "has_method", {}),
        ("entry", "method", "calls", {"weight": 1}),
        ("handler", "entry", "calls", {}),
        ("route", "handler", "handled_by", {}),
        ("method", "database", "depends_on", {}),
        ("noise", "entry", "mentions", {}),
    ]
    return nodes, edges


def _graph_patch(*, reverse=False):
    nodes, edges = _code_graph()
    if reverse:
        nodes = list(reversed(nodes))
        edges = list(reversed(edges))
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    return engine, patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


@pytest.mark.asyncio
async def test_query_facts_uses_exact_graph_filters_and_raw_fact_properties():
    engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "query_facts",
            "kind": "symbol",
            "file_prefix": "pkg/",
            "relation": "has_method",
            "prop": "exported",
            "prop_value": "true",
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("")

    assert result["total"] == 1
    assert result["facts"] == [
        {
            "id": str(WIDGET_ID),
            "kind": "symbol",
            "type": "CodeSymbol",
            "name": "pkg.Widget",
            "file": "pkg/widget.py",
            "line": 10,
            "repo": "repo-a",
            "symbol_kind": "class",
            "properties": {"exported": True, "language": "python"},
            "relations": [
                {"type": "has_method", "target_id": "method", "target": "pkg.Widget.run"}
            ],
        }
    ]
    engine.get_filtered_graph_data.assert_awaited_once_with(
        [{"type": [*CODE_NODE_TYPES, "CodeRepository"]}]
    )
    # Repository nodes are fetched for the delta operation but stay out of the
    # fact indexes and every fact-facing operation.
    assert "CodeRepository" not in CODE_NODE_TYPES


def test_query_facts_builds_an_exact_property_index():
    nodes, edges = _code_graph()
    graph = _CodeGraphSnapshot(nodes, edges)

    assert graph.by_property["exported"]["true"] == {str(WIDGET_ID)}
    assert graph.by_property["language"]["python"] == {str(WIDGET_ID)}


@pytest.mark.asyncio
async def test_query_facts_is_stably_sorted_and_paginated():
    _engine, graph_patch = _graph_patch(reverse=True)
    retriever = CodeRetriever(
        config={"operation": "query_facts", "kind": "symbol", "offset": 1, "limit": 2}
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("")

    assert result["total"] == 7
    assert result["has_more"] is True
    assert [fact["name"] for fact in result["facts"]] == ["pkg.NewWidget", "pkg.Widget"]


@pytest.mark.asyncio
async def test_query_facts_structured_filters_ignore_generic_nonempty_query_text():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(config={"operation": "query_facts", "kind": "symbol"})

    with graph_patch:
        result = await retriever.get_retrieved_objects("What is in the document?")

    assert result["total"] == 7

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        page = await CodeRetriever(
            config={"operation": "query_facts", "limit": 2}
        ).get_retrieved_objects("What is in the document?")

    assert page["total"] == 9
    assert len(page["facts"]) == 2


@pytest.mark.asyncio
async def test_explore_rejects_ambiguous_name_and_repo_disambiguates():
    _engine, graph_patch = _graph_patch()
    with graph_patch:
        with pytest.raises(CodeSearchValidationError, match="ambiguous") as error:
            await CodeRetriever(config={"operation": "explore"}).get_retrieved_objects(
                "shared.Helper"
            )
    assert error.value.status_code == 422

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        result = await CodeRetriever(
            config={"operation": "explore", "repo": "repo-b"}
        ).get_retrieved_objects("shared.Helper")

    assert result["focus"]["id"] == "helper-b"


@pytest.mark.asyncio
async def test_explore_is_bidirectional_and_clamps_depth_to_two():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={"operation": "explore", "name": "pkg.Widget.run", "max_depth": 99}
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("ignored")

    depths = {node["name"]: node["depth"] for node in result["nodes"]}
    assert depths["pkg.Widget"] == 1
    assert depths["pkg.entry"] == 1
    assert depths["widget_db"] == 1
    assert depths["api.handler"] == 2
    assert result["stats"]["max_depth_reached"] == 2


@pytest.mark.asyncio
async def test_traverse_supports_reverse_bfs_and_relation_filter():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "traverse",
            "direction": "reverse",
            "relation_kinds": ["calls"],
            "max_depth": 2,
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("pkg.Widget.run")

    assert [(node["name"], node["depth"]) for node in result["nodes"]] == [
        ("pkg.Widget.run", 0),
        ("pkg.entry", 1),
        ("api.handler", 2),
    ]
    assert all("relations" not in node for node in result["nodes"])
    assert [edge["type"] for edge in result["edges"]] == ["calls", "calls"]


@pytest.mark.asyncio
async def test_reverse_traverse_rolls_up_type_members_and_constructor():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "traverse",
            "direction": "reverse",
            "relation_types": ["calls"],
            "max_depth": 2,
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("pkg.Widget")

    depths = {node["name"]: node["depth"] for node in result["nodes"]}
    assert depths["pkg.Widget"] == 0
    assert depths["pkg.Widget.run"] == 0
    assert depths["pkg.NewWidget"] == 0
    assert depths["pkg.entry"] == 1
    assert depths["api.handler"] == 2


@pytest.mark.asyncio
async def test_find_path_returns_stable_shortest_forward_path():
    _engine, graph_patch = _graph_patch(reverse=True)
    retriever = CodeRetriever(
        config={
            "operation": "find_path",
            "target": "pkg.Widget.run",
            "relation_types": ["calls"],
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("api.handler")

    assert result["found"] is True
    assert [node["name"] for node in result["path"]] == [
        "api.handler",
        "pkg.entry",
        "pkg.Widget.run",
    ]
    assert [node["depth"] for node in result["path"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_find_path_rolls_up_type_target_to_its_methods():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "find_path",
            "target": "pkg.Widget",
            "relation_types": ["calls"],
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("api.handler")

    assert result["found"] is True
    assert result["to"]["name"] == "pkg.Widget"
    assert result["matched_to"]["name"] == "pkg.Widget.run"
    assert [node["name"] for node in result["path"]] == [
        "api.handler",
        "pkg.entry",
        "pkg.Widget.run",
    ]


@pytest.mark.asyncio
async def test_enola_operation_seed_aliases_are_supported():
    _engine, graph_patch = _graph_patch()
    with graph_patch:
        explored = await CodeRetriever(
            config={"operation": "explore", "focus": "pkg.Widget.run"}
        ).get_retrieved_objects("")
    assert explored["focus"]["name"] == "pkg.Widget.run"

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        traversed = await CodeRetriever(
            config={"operation": "traverse", "start": "api.handler", "max_depth": 1}
        ).get_retrieved_objects("")
    assert traversed["nodes"][0]["name"] == "api.handler"

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        path = await CodeRetriever(
            config={
                "operation": "find_path",
                "from": "api.handler",
                "to": "pkg.Widget.run",
            }
        ).get_retrieved_objects("")
    assert path["found"] is True

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        impact = await CodeRetriever(
            config={"operation": "impact_analysis", "target": "pkg.Widget"}
        ).get_retrieved_objects("")
    assert impact["targets"][0]["name"] == "pkg.Widget"


@pytest.mark.asyncio
async def test_impact_rolls_up_type_members_and_counts_beyond_display_cap():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "impact_analysis",
            "max_depth": 2,
            # Three depth-zero rollup seeds + one displayed dependent.
            "max_nodes": 4,
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("pkg.Widget")

    assert {seed["name"] for seed in result["impact_seeds"]} == {
        "pkg.Widget",
        "pkg.Widget.run",
        "pkg.NewWidget",
    }
    assert result["total_dependents"] == 2
    assert result["stats"]["truncated"] is True
    assert [node["name"] for node in result["by_depth"]["1"]] == ["pkg.entry"]
    assert "2 total dependents (showing 1)" in result["summary"]


@pytest.mark.asyncio
async def test_impact_reports_cross_repo_and_optional_forward_dependencies():
    _engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(
        config={
            "operation": "impact_analysis",
            "max_depth": 3,
            "include_forward": True,
        }
    )

    with graph_patch:
        result = await retriever.get_retrieved_objects("pkg.Widget")

    assert result["cross_repo_impact"] == ["repo-b"]
    assert "forward_dependencies" in result
    assert result["forward_dependencies"]["nodes"][0]["name"] == "pkg.Widget"


@pytest.mark.asyncio
async def test_impact_rollup_finds_bare_type_constructor_in_same_package():
    nodes = [
        (
            "bare-type",
            {
                "name": "Widget",
                "type": "CodeSymbol",
                "file_path": "pkg/widget.py",
                "repo": "repo-a",
                "symbol_kind": "class",
            },
        ),
        (
            "bare-constructor",
            {
                "name": "NewWidget",
                "type": "CodeSymbol",
                "file_path": "pkg/factory.py",
                "repo": "repo-a",
                "symbol_kind": "function",
            },
        ),
        (
            "wrong-package",
            {
                "name": "NewWidget",
                "type": "CodeSymbol",
                "file_path": "other/factory.py",
                "repo": "repo-a",
                "symbol_kind": "function",
            },
        ),
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, []))
    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        result = await CodeRetriever(config={"operation": "impact_analysis"}).get_retrieved_objects(
            "Widget"
        )

    assert {seed["id"] for seed in result["impact_seeds"]} == {
        "bare-type",
        "bare-constructor",
    }


@pytest.mark.asyncio
async def test_code_retriever_never_calls_llm_embeddings_or_session_analysis():
    engine, graph_patch = _graph_patch(reverse=True)
    llm_call = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    vector_call = MagicMock(side_effect=AssertionError("vector engine must not be called"))
    retriever = CodeRetriever(config={"operation": "explore", "repo": "repo-a"})

    with (
        graph_patch,
        patch("cognee.modules.retrieval.utils.completion.generate_completion", llm_call),
        patch("cognee.infrastructure.databases.vector.get_vector_engine", vector_call),
    ):
        preparation = await retriever.prepare_session_turn_for_retrieval("pkg.Widget")
        result = await retriever.get_retrieved_objects("pkg.Widget")
        context = await retriever.get_context_from_objects(
            query="pkg.Widget", retrieved_objects=result
        )
        completion = await retriever.get_completion_from_context(
            query="pkg.Widget", retrieved_objects=result, context=context
        )

    assert retriever.supports_session_turn_preparation is False
    assert preparation.effective_query == "pkg.Widget"
    assert json.loads(context) == result
    assert context == json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert completion is result
    llm_call.assert_not_awaited()
    vector_call.assert_not_called()
    engine.get_filtered_graph_data.assert_awaited_once()


def test_invalid_operation_and_arguments_raise_clear_errors():
    with pytest.raises(CodeSearchValidationError, match="Unsupported CODE operation") as error:
        CodeRetriever(config={"operation": "guess"})
    assert error.value.status_code == 422

    with pytest.raises(CodeSearchValidationError, match="direction"):
        CodeRetriever(config={"operation": "traverse", "direction": "sideways"})._traverse(
            MagicMock(), "seed"
        )


@pytest.mark.asyncio
async def test_graph_snapshot_indexes_are_reused_across_queries():
    engine, graph_patch = _graph_patch()

    with graph_patch:
        facts = await CodeRetriever(
            config={"operation": "query_facts", "kind": "symbol"}
        ).get_retrieved_objects("")
        explored = await CodeRetriever(
            config={"operation": "explore", "repo": "repo-a"}
        ).get_retrieved_objects("pkg.Widget")

    assert facts["total"] == 7
    assert explored["focus"]["name"] == "pkg.Widget"
    engine.get_filtered_graph_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_cached_snapshot_results_cannot_mutate_later_queries():
    engine, graph_patch = _graph_patch()

    with graph_patch:
        first = await CodeRetriever(
            config={"operation": "query_facts", "name": "pkg.Widget"}
        ).get_retrieved_objects("")
        first["facts"][0]["properties"]["language"] = "mutated"
        second = await CodeRetriever(
            config={"operation": "query_facts", "name": "pkg.Widget"}
        ).get_retrieved_objects("")

    assert second["facts"][0]["properties"]["language"] == "python"
    engine.get_filtered_graph_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_snapshot_cache_isolated_by_dataset_and_database():
    engine, graph_patch = _graph_patch()
    graph_config = {
        "graph_database_provider": "ladybug",
        "graph_file_path": "/tmp/graph-a",
        "graph_database_name": "graph-a",
    }
    dataset_token = current_dataset_id.set("dataset-a")

    try:
        with (
            graph_patch,
            patch(
                "cognee.modules.retrieval.code_retriever.get_graph_context_config",
                return_value=graph_config,
            ) as config_mock,
        ):
            await CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")

            current_dataset_id.set("dataset-b")
            await CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")

            current_dataset_id.set("dataset-a")
            config_mock.return_value = {**graph_config, "graph_file_path": "/tmp/graph-b"}
            await CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")

            config_mock.return_value = graph_config
            await CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")
    finally:
        current_dataset_id.reset(dataset_token)

    assert engine.get_filtered_graph_data.await_count == 3


@pytest.mark.asyncio
async def test_graph_snapshot_cache_invalidation_reloads_indexes():
    nodes, edges = _code_graph()
    changed_nodes = [
        *nodes,
        (
            "new-symbol",
            {
                "name": "pkg.new_symbol",
                "type": "CodeSymbol",
                "file_path": "pkg/new.py",
                "repo": "repo-a",
            },
        ),
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(side_effect=[(nodes, edges), (changed_nodes, edges)])

    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        initial = await CodeRetriever(
            config={"operation": "query_facts", "kind": "symbol"}
        ).get_retrieved_objects("")
        cached = await CodeRetriever(
            config={"operation": "query_facts", "kind": "symbol"}
        ).get_retrieved_objects("")
        invalidate_code_graph_snapshot_cache()
        refreshed = await CodeRetriever(
            config={"operation": "query_facts", "kind": "symbol"}
        ).get_retrieved_objects("")

    assert initial["total"] == cached["total"] == 7
    assert refreshed["total"] == 8
    assert engine.get_filtered_graph_data.await_count == 2


@pytest.mark.asyncio
async def test_graph_snapshot_cache_is_bounded_lru_and_expires():
    now = [0.0]
    cache = _CodeGraphSnapshotCache(max_entries=2, ttl_seconds=5.0, clock=lambda: now[0])
    config = {"graph_database_provider": "ladybug"}
    keys = [
        _code_graph_snapshot_cache_key(dataset_id=f"dataset-{index}", graph_config=config)
        for index in range(3)
    ]
    loads = 0

    async def load():
        nonlocal loads
        loads += 1
        return _CodeGraphSnapshot([], [])

    first = await cache.get_or_load(keys[0], load)
    await cache.get_or_load(keys[1], load)
    assert await cache.get_or_load(keys[0], load) is first

    await cache.get_or_load(keys[2], load)
    assert len(cache) == 2
    assert cache.contains(keys[0])
    assert not cache.contains(keys[1])

    now[0] = 6.0
    assert await cache.get_or_load(keys[0], load) is not first
    assert loads == 4


@pytest.mark.asyncio
async def test_graph_snapshot_load_is_single_flight_for_concurrent_queries():
    nodes, edges = _code_graph()
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    calls = 0

    async def load_graph(_filters):
        nonlocal calls
        calls += 1
        load_started.set()
        await release_load.wait()
        return nodes, edges

    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(side_effect=load_graph)

    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        first = asyncio.create_task(
            CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")
        )
        await load_started.wait()
        second = asyncio.create_task(
            CodeRetriever(config={"operation": "query_facts"}).get_retrieved_objects("")
        )
        await asyncio.sleep(0)
        release_load.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result == second_result
    assert calls == 1


@pytest.mark.asyncio
async def test_invalidation_during_load_discards_stale_snapshot():
    old_nodes, edges = _code_graph()
    new_nodes = [
        *old_nodes,
        (
            "new-symbol",
            {"name": "new_symbol", "type": "CodeSymbol", "repo": "repo-a"},
        ),
    ]
    first_load_started = asyncio.Event()
    release_first_load = asyncio.Event()
    calls = 0

    async def load_graph(_filters):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_load_started.set()
            await release_first_load.wait()
            return old_nodes, edges
        return new_nodes, edges

    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(side_effect=load_graph)

    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        first = asyncio.create_task(
            CodeRetriever(
                config={"operation": "query_facts", "kind": "symbol"}
            ).get_retrieved_objects("")
        )
        await first_load_started.wait()
        invalidate_code_graph_snapshot_cache()
        second = asyncio.create_task(
            CodeRetriever(
                config={"operation": "query_facts", "kind": "symbol"}
            ).get_retrieved_objects("")
        )
        second_result = await second
        release_first_load.set()
        first_result = await first

    assert first_result["total"] == second_result["total"] == 8
    assert calls == 2


@pytest.mark.asyncio
async def test_delta_operation_reports_repository_last_delta():
    engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(config={"operation": "delta"})

    with graph_patch:
        result = await retriever.get_retrieved_objects("")

    assert result["operation"] == "delta"
    assert [repo["repo"] for repo in result["repositories"]] == ["repo-a"]
    repository = result["repositories"][0]
    # The fixture repository node predates delta stamping.
    assert repository["delta"] is None
    assert repository["last_snapshot_id"] is None


@pytest.mark.asyncio
async def test_delta_operation_repo_filter_and_stamped_payload():
    nodes, edges = _code_graph()
    delta = {"nodes_added": 2, "nodes_removed": 1, "edges_added": 3, "edges_removed": 0}
    nodes = list(nodes) + [
        (
            "repo-b",
            {
                "name": "repo-b",
                "type": "CodeRepository",
                "path": "/src/repo-b",
                "last_snapshot_id": "sha256:abc",
                "last_delta": delta,
            },
        )
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    graph_patch = patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )
    retriever = CodeRetriever(config={"operation": "delta", "repo": "repo-b"})

    with graph_patch:
        result = await retriever.get_retrieved_objects("")

    assert [repo["repo"] for repo in result["repositories"]] == ["repo-b"]
    assert result["repositories"][0]["last_snapshot_id"] == "sha256:abc"
    assert result["repositories"][0]["delta"] == delta


@pytest.mark.asyncio
async def test_repository_nodes_stay_out_of_fact_operations():
    engine, graph_patch = _graph_patch()
    retriever = CodeRetriever(config={"operation": "query_facts", "limit": 100})

    with graph_patch:
        result = await retriever.get_retrieved_objects("")

    assert all(fact["type"] != "CodeRepository" for fact in result["facts"])


# --- enola 0.4.x: insights, writer ids, receipt, new kinds ---------------------

ENOLA_ID_DB = "a" * 32
ENOLA_ID_API = "b" * 32


def _insight_graph():
    nodes = [
        (
            "repo-a",
            {
                "name": "repo-a",
                "type": "CodeRepository",
                "path": "/src/repo-a",
                "last_snapshot_id": "sha256:abc",
                "last_receipt": {
                    "format_version": 1,
                    "enola_version": "0.4.12",
                    "quality": {"files_seen": 6, "files_parsed": 5, "parse_errors": 0},
                },
            },
        ),
        (
            "db",
            {
                "name": "app/db.Database",
                "type": "CodeSymbol",
                "file_path": "app/db.py",
                "line": 4,
                "end_line": 40,
                "repo": "repo-a",
                "symbol_kind": "class",
                "enola_id": ENOLA_ID_DB,
            },
        ),
        (
            "api",
            {
                "name": "app/api.handler",
                "type": "CodeSymbol",
                "file_path": "app/api.py",
                "repo": "repo-a",
                "symbol_kind": "function",
                "enola_id": ENOLA_ID_API,
            },
        ),
        (
            "hot",
            {
                "name": "Call-graph hotspot: app/db.Database",
                "type": "CodeInsight",
                "repo": "repo-a",
                "description": "A pinch point.",
                "fact_properties": {
                    "source": "hotspots",
                    "confidence": 0.7,
                    "metrics": {"fan_in": 9},
                },
            },
        ),
        (
            "cyc",
            {
                "name": "Dependency cycle: app/api -> app/db -> app/api",
                "type": "CodeInsight",
                "repo": "repo-a",
                "fact_properties": {"source": "cycles", "confidence": 1.0},
            },
        ),
        (
            "info",
            {
                "name": "Domain findings do not apply to this repository",
                "type": "CodeInsight",
                "repo": "repo-a",
                "fact_properties": {"source": "domain", "confidence": 1.0, "informational": True},
            },
        ),
        (
            "extraction",
            {
                "name": "python:calls",
                "type": "CodeExtractionAccount",
                "repo": "repo-a",
                "fact_properties": {"extractor": "python", "language": "python"},
            },
        ),
    ]
    edges = [
        ("hot", "db", "evidences", {}),
        ("cyc", "api", "evidences", {}),
        ("cyc", "db", "evidences", {}),
        ("api", "db", "calls", {}),
    ]
    return nodes, edges


def _insight_graph_patch():
    nodes, edges = _insight_graph()
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    return patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


@pytest.mark.asyncio
async def test_insights_operation_filters_by_confidence_and_resolves_evidence():
    retriever = CodeRetriever(
        config={"operation": "insights", "min_confidence": 0.9, "informational": False}
    )

    with _insight_graph_patch():
        result = await retriever.get_retrieved_objects("")

    assert result["operation"] == "insights"
    assert result["total"] == 1
    assert result["by_source"] == {"cycles": 1}
    (cycle,) = result["insights"]
    assert cycle["kind"] == "insight"
    assert cycle["name"].startswith("Dependency cycle")
    assert cycle["properties"] == {"source": "cycles", "confidence": 1.0}
    assert [fact["name"] for fact in cycle["evidence"]] == ["app/api.handler", "app/db.Database"]
    assert cycle["evidence"][1]["enola_id"] == ENOLA_ID_DB
    assert cycle["evidence"][1]["end_line"] == 40
    assert "relations" not in cycle


@pytest.mark.asyncio
async def test_insights_operation_orders_by_confidence_then_name_and_filters_source():
    with _insight_graph_patch():
        everything = await CodeRetriever(config={"operation": "insights"}).get_retrieved_objects("")

    assert everything["total"] == 3
    assert [insight["id"] for insight in everything["insights"]] == ["cyc", "info", "hot"]
    assert everything["by_source"] == {"cycles": 1, "domain": 1, "hotspots": 1}

    with _insight_graph_patch():
        hotspots = await CodeRetriever(
            config={"operation": "insights", "source": "hotspots"}
        ).get_retrieved_objects("")

    assert [insight["id"] for insight in hotspots["insights"]] == ["hot"]
    assert hotspots["insights"][0]["properties"]["metrics"] == {"fan_in": 9}
    assert hotspots["insights"][0]["description"] == "A pinch point."

    # Like query_facts: the query text is a title substring only for an
    # otherwise unconfigured call, and never narrows a structured one.
    with _insight_graph_patch():
        by_title = await CodeRetriever(config={"operation": "insights"}).get_retrieved_objects(
            "cycle"
        )
    assert [insight["id"] for insight in by_title["insights"]] == ["cyc"]

    with _insight_graph_patch():
        structured = await CodeRetriever(
            config={"operation": "insights", "limit": 10}
        ).get_retrieved_objects("What did the explainers find?")
    assert structured["total"] == 3


@pytest.mark.asyncio
async def test_insights_operation_paginates_and_validates_arguments():
    with _insight_graph_patch():
        page = await CodeRetriever(
            config={"operation": "insights", "offset": 1, "limit": 1}
        ).get_retrieved_objects("")

    assert page["total"] == 3
    assert page["has_more"] is True
    assert [insight["id"] for insight in page["insights"]] == ["info"]

    for bad in ({"min_confidence": 2}, {"min_confidence": "high"}, {"informational": "maybe"}):
        with pytest.raises(CodeSearchValidationError):
            with _insight_graph_patch():
                await CodeRetriever(config={"operation": "insights", **bad}).get_retrieved_objects(
                    ""
                )


@pytest.mark.asyncio
async def test_operations_accept_enola_fact_ids_as_seeds():
    with _insight_graph_patch():
        explored = await CodeRetriever(
            config={"operation": "explore", "id": ENOLA_ID_DB}
        ).get_retrieved_objects("")

    assert explored["focus"]["id"] == "db"
    assert explored["focus"]["enola_id"] == ENOLA_ID_DB
    assert explored["focus"]["name"] == "app/db.Database"

    with _insight_graph_patch():
        path = await CodeRetriever(
            config={"operation": "find_path", "source_id": ENOLA_ID_API, "target_id": ENOLA_ID_DB}
        ).get_retrieved_objects("")

    assert path["found"] is True
    assert [node["id"] for node in path["path"]] == ["api", "db"]

    with pytest.raises(CodeSearchValidationError, match="could not resolve"):
        with _insight_graph_patch():
            await CodeRetriever(
                config={"operation": "explore", "id": "f" * 32}
            ).get_retrieved_objects("")


@pytest.mark.asyncio
async def test_delta_operation_reports_the_stamped_receipt():
    with _insight_graph_patch():
        result = await CodeRetriever(config={"operation": "delta"}).get_retrieved_objects("")

    (repository,) = result["repositories"]
    assert repository["last_snapshot_id"] == "sha256:abc"
    assert repository["receipt"]["format_version"] == 1
    assert repository["receipt"]["enola_version"] == "0.4.12"
    assert repository["receipt"]["quality"]["files_parsed"] == 5


@pytest.mark.asyncio
async def test_new_fact_kinds_are_queryable():
    with _insight_graph_patch():
        result = await CodeRetriever(
            config={"operation": "query_facts", "kind": "extraction"}
        ).get_retrieved_objects("")

    assert result["total"] == 1
    assert result["facts"][0]["kind"] == "extraction"
    assert result["facts"][0]["type"] == "CodeExtractionAccount"
    assert result["facts"][0]["properties"]["extractor"] == "python"


# --- diagrams and the architecture overview ------------------------------------


@pytest.mark.asyncio
async def test_diagram_option_attaches_mermaid_source_to_any_operation():
    _engine, graph_patch = _graph_patch()
    with graph_patch:
        result = await CodeRetriever(
            config={"operation": "explore", "name": "pkg.Widget.run", "diagram": True}
        ).get_retrieved_objects("")

    diagram = result["diagram"]
    assert diagram["format"] == "mermaid"
    assert diagram["nodes"] == len(result["nodes"])
    assert diagram["edges"] == len(result["edges"])
    source = diagram["source"]
    assert source.startswith('---\ntitle: "explore: pkg.Widget.run"\n---\nflowchart LR\n')
    # Every node is declared with a kind-specific shape and a quoted label...
    assert '(["pkg.Widget.run"])' in source
    assert '[("widget_db")]' in source
    # ...edges carry the relation name, and the focus is highlighted.
    assert '-- "has_method" -->' in source
    assert "classDef highlight" in source
    assert "\n    class n" in source
    # Deterministic: rendering the same result twice is byte-identical.
    _engine, graph_patch = _graph_patch(reverse=True)
    with graph_patch:
        again = await CodeRetriever(
            config={"operation": "explore", "name": "pkg.Widget.run", "diagram": "mermaid"}
        ).get_retrieved_objects("")
    assert again["diagram"]["source"] == source


@pytest.mark.asyncio
async def test_diagram_groups_repositories_and_renders_dot():
    _engine, graph_patch = _graph_patch()
    with graph_patch:
        result = await CodeRetriever(
            config={
                "operation": "impact_analysis",
                "name": "pkg.Widget.run",
                "max_depth": 5,
                "diagram": "dot",
            }
        ).get_retrieved_objects("")

    source = result["diagram"]["source"]
    assert result["diagram"]["format"] == "dot"
    assert source.startswith("digraph code_graph {")
    # repo-a and repo-b both appear -> one cluster per repository.
    assert 'label="repo-a";' in source and 'label="repo-b";' in source
    assert "subgraph cluster_0 {" in source and "subgraph cluster_1 {" in source
    assert '[label="calls"];' in source
    # The impact target is highlighted.
    assert "penwidth=3" in source

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        path = await CodeRetriever(
            config={
                "operation": "find_path",
                "source": "api.handler",
                "target": "widget_db",
                "diagram": "mermaid",
            }
        ).get_retrieved_objects("")
    assert path["found"] is True
    assert path["diagram"]["nodes"] == len(path["path"])
    assert path["diagram"]["source"].count("-->") == len(path["edges"])


@pytest.mark.asyncio
async def test_diagram_option_is_validated_and_off_by_default():
    _engine, graph_patch = _graph_patch()
    with graph_patch:
        plain = await CodeRetriever(config={"operation": "explore"}).get_retrieved_objects(
            "pkg.Widget.run"
        )
    assert "diagram" not in plain

    with pytest.raises(CodeSearchValidationError, match="diagram"):
        CodeRetriever(config={"operation": "explore", "diagram": "png"})

    _engine, graph_patch = _graph_patch()
    with graph_patch:
        delta = await CodeRetriever(
            config={"operation": "delta", "diagram": True}
        ).get_retrieved_objects("")
    assert delta["diagram"]["source"] is None
    assert delta["diagram"]["nodes"] == 0
    assert "note" in delta["diagram"]


def test_diagram_labels_are_escaped_for_both_formats():
    from cognee.modules.retrieval.code_graph_diagram import render_dot, render_mermaid

    nodes = [
        {"id": "a", "kind": "route", "name": 'GET /items?q="x"<y>#1&2', "repo": "r"},
        {"id": "b", "kind": "module", "name": ".", "repo": "r"},
    ]
    edges = [{"source_id": "a", "target_id": "b", "type": "declares", "count": 3}]

    mermaid = render_mermaid(nodes, edges, title='a "quoted" # title: Vec<T>')
    # The front-matter title is YAML (quoted; angle brackets swapped for
    # guillemets, which Mermaid draws verbatim), the labels use entity codes.
    assert mermaid.startswith('---\ntitle: "a \\"quoted\\" # title: Vec\u2039T\u203a"\n---\n')
    assert 'n0>"GET /items?q=#quot;x#quot;#lt;y#gt;#35;1#38;2"]' in mermaid
    assert 'n1[["(root)"]]' in mermaid
    assert 'n0 -- "declares x3" --> n1' in mermaid
    assert "classDef" not in mermaid

    dot = render_dot(nodes, edges, highlight_ids=["b"])
    assert 'n0 [label="GET /items?q=\\"x\\"<y>#1&2" shape=cds];' in dot
    assert 'n1 [label="(root)" shape=component color="#d64545" penwidth=3];' in dot
    assert 'n0 -> n1 [label="declares x3"];' in dot


def _architecture_graph():
    nodes = [
        ("repo", {"name": "shop", "type": "CodeRepository", "path": "/src/shop"}),
        ("m_root", {"name": ".", "type": "CodeModule", "file_path": ".", "repo": "shop"}),
        (
            "m_inv",
            {"name": "inventory", "type": "CodeModule", "file_path": "inventory", "repo": "shop"},
        ),
        ("m_api", {"name": "api", "type": "CodeModule", "file_path": "api", "repo": "shop"}),
        (
            "s_main",
            {
                "name": "main.main",
                "type": "CodeSymbol",
                "file_path": "main.py",
                "repo": "shop",
                "symbol_kind": "function",
            },
        ),
        (
            "s_total",
            {
                "name": "inventory/pricing.total",
                "type": "CodeSymbol",
                "file_path": "inventory/pricing.py",
                "repo": "shop",
                "symbol_kind": "function",
            },
        ),
        (
            "s_line",
            {
                "name": "inventory/pricing.line",
                "type": "CodeSymbol",
                "file_path": "inventory/pricing.py",
                "repo": "shop",
                "symbol_kind": "function",
            },
        ),
        (
            "s_handler",
            {
                "name": "api/routes.orders",
                "type": "CodeSymbol",
                "file_path": "api/routes.py",
                "repo": "shop",
                "symbol_kind": "function",
            },
        ),
        (
            "route",
            {
                "name": "GET /orders",
                "type": "ApiEndpoint",
                "file_path": "api/routes.py",
                "repo": "shop",
            },
        ),
        (
            "db",
            {
                "name": "orders_table",
                "type": "StorageResource",
                "file_path": "inventory/store.py",
                "repo": "shop",
            },
        ),
        (
            "dep",
            {
                "name": "main -> requests",
                "type": "ExternalDependency",
                "file_path": "main.py",
                "repo": "shop",
            },
        ),
    ]
    edges = [
        ("s_main", "m_root", "declares", {}),
        ("s_total", "m_inv", "declares", {}),
        ("s_line", "m_inv", "declares", {}),
        ("s_handler", "m_api", "declares", {}),
        ("route", "m_api", "declares", {}),
        ("db", "m_inv", "declares", {}),
        # cross-module calls roll up to module edges with counts
        ("s_main", "s_total", "calls", {}),
        ("s_main", "s_line", "calls", {}),
        ("s_handler", "s_total", "calls", {}),
        # intra-module call: dropped
        ("s_total", "s_line", "calls", {}),
        ("route", "s_handler", "handled_by", {}),
        ("s_total", "db", "uses", {}),
        ("dep", "m_root", "declares", {}),
        # containment edges are never drawn
        ("s_main", "repo", "part_of", {}),
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    return patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


@pytest.mark.asyncio
async def test_architecture_rolls_symbol_edges_up_to_modules_and_draws_by_default():
    with _architecture_graph():
        result = await CodeRetriever(config={"operation": "architecture"}).get_retrieved_objects("")

    assert result["operation"] == "architecture"
    assert result["repos"] == ["shop"]
    assert [node["name"] for node in result["nodes"]] == [
        "GET /orders",
        ".",
        "api",
        "inventory",
        "orders_table",
    ]
    assert [(e["source"], e["target"], e["type"], e["count"]) for e in result["edges"]] == [
        (".", "inventory", "calls", 2),
        ("api", "inventory", "calls", 1),
        ("api", "GET /orders", "declares", 1),
        ("inventory", "orders_table", "declares", 1),
        ("GET /orders", "api", "handled_by", 1),
        ("inventory", "orders_table", "uses", 1),
    ]
    assert result["stats"] == {
        "nodes_total": 5,
        "nodes_shown": 5,
        "edges_shown": 6,
        "truncated": False,
    }
    diagram = result["diagram"]
    assert diagram["format"] == "mermaid"
    assert 'title: "architecture: shop"' in diagram["source"]
    assert '[["(root)"]]' in diagram["source"]
    assert '-- "calls x2" -->' in diagram["source"]
    assert '>"GET /orders"]' in diagram["source"]


@pytest.mark.asyncio
async def test_architecture_respects_filters_bounds_and_opt_out():
    with _architecture_graph():
        modules_only = await CodeRetriever(
            config={
                "operation": "architecture",
                "node_types": ["module"],
                "relation_types": ["calls"],
                "max_nodes": 2,
                "diagram": False,
            }
        ).get_retrieved_objects("")

    assert "diagram" not in modules_only
    # Best-connected modules survive the cap: "." and "inventory" (api is dropped).
    assert [node["name"] for node in modules_only["nodes"]] == [".", "inventory"]
    assert [(e["source"], e["target"], e["count"]) for e in modules_only["edges"]] == [
        (".", "inventory", 2)
    ]
    assert modules_only["stats"]["truncated"] is True

    with _architecture_graph():
        other_repo = await CodeRetriever(
            config={"operation": "architecture", "repo": "elsewhere", "diagram": "dot"}
        ).get_retrieved_objects("")
    assert other_repo["nodes"] == [] and other_repo["edges"] == []
    assert other_repo["diagram"]["source"].startswith("digraph code_graph {")
