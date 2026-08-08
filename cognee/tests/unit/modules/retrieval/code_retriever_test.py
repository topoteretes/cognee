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
    engine.get_filtered_graph_data.assert_awaited_once_with([{"type": list(CODE_NODE_TYPES)}])
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
