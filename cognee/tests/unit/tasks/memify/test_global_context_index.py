from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.memify.global_context_index.constants import SUMMARIZED_IN
from cognee.tasks.memify.global_context_index.load import (
    extract_context_index_input,
)
from cognee.tasks.memify.global_context_index.models import (
    GlobalContextIndexInput,
    GlobalContextIndexUpdateData,
    SummaryNode,
)
from cognee.tasks.memify.global_context_index.persist import (
    ensure_global_context_storage_context,
)
from cognee.tasks.memify.global_context_index.update import (
    update_global_context_index,
)
from cognee.tasks.summarization.models import GlobalContextSummary

update_global_context_index_module = import_module(
    "cognee.tasks.memify.global_context_index.update"
)
load_module = import_module("cognee.tasks.memify.global_context_index.load")
summary_generation_module = import_module("cognee.tasks.memify.global_context_index.summarize")


def _summary_node(text: str = "summary", bucket_id: str | None = None) -> SummaryNode:
    return SummaryNode(
        id=str(uuid4()),
        text=text,
        type="TextSummary",
        global_context_bucket_id=bucket_id,
    )


def _text_summary_graph_node(bucket_id: str | None = None) -> Node:
    attributes = {"type": "TextSummary", "text": "Chunk summary"}
    if bucket_id is not None:
        attributes["global_context_bucket_id"] = bucket_id
    return Node(str(uuid4()), attributes)


def _global_context_summary_graph_node(
    dataset_id: str,
    *,
    node_id: str | None = None,
    level: int | None = 0,
    is_root: bool = False,
    bucket_id: str | None = None,
    graph_bucket_entity_ids: list[str] | None = None,
    include_graph_bucket_entity_ids: bool = False,
) -> Node:
    attributes = {
        "type": "GlobalContextSummary",
        "text": "Global context summary",
        "dataset_id": dataset_id,
        "level": level,
        "is_root": is_root,
    }
    if bucket_id is not None:
        attributes["global_context_bucket_id"] = bucket_id
    if include_graph_bucket_entity_ids:
        attributes["graph_bucket_entity_ids"] = graph_bucket_entity_ids
    return Node(node_id or str(uuid4()), attributes)


def _add_summarized_in_edge(graph: CogneeGraph, child: Node, parent: Node) -> None:
    graph.add_edge(
        Edge(
            child,
            parent,
            attributes={"relationship_name": SUMMARIZED_IN},
        )
    )


def _assert_add_data_points_used_context(add_data_points_mock: AsyncMock, ctx) -> None:
    assert add_data_points_mock.await_args_list
    assert all(call.kwargs.get("ctx") is ctx for call in add_data_points_mock.await_args_list)


def test_ensure_global_context_storage_context_adds_stable_data_item_id():
    dataset_id = uuid4()
    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=dataset_id),
        data_item={},
    )

    result = ensure_global_context_storage_context(ctx)

    assert result is ctx
    assert ctx.data_item.id == uuid5(
        NAMESPACE_URL,
        f"cognee:global-context-index:{dataset_id}",
    )


def test_global_context_index_package_exports_pipeline_tasks():
    package_module = import_module("cognee.tasks.memify.global_context_index")
    update_module = import_module("cognee.tasks.memify.global_context_index.update")

    assert package_module.update_global_context_index is update_module.update_global_context_index
    assert (
        package_module.extract_global_context_index_input
        is load_module.extract_global_context_index_input
    )
    assert GlobalContextIndexInput is GlobalContextIndexUpdateData


@pytest.mark.asyncio
async def test_extract_global_context_index_input_loads_graph_only_for_missing_data(monkeypatch):
    graph = CogneeGraph()
    child = _text_summary_graph_node()
    graph.add_node(child)
    get_fragment_mock = AsyncMock(return_value=graph)
    monkeypatch.setattr(load_module, "get_context_index_memory_fragment", get_fragment_mock)

    result = await load_module.extract_global_context_index_input(None)

    assert [summary.id for summary in result.text_summaries] == [child.id]
    get_fragment_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_global_context_index_input_loads_graph_for_empty_dict_sentinel(
    monkeypatch,
):
    graph = CogneeGraph()
    child = _text_summary_graph_node()
    graph.add_node(child)
    get_fragment_mock = AsyncMock(return_value=graph)
    monkeypatch.setattr(load_module, "get_context_index_memory_fragment", get_fragment_mock)

    result = await load_module.extract_global_context_index_input({})

    assert [summary.id for summary in result.text_summaries] == [child.id]
    get_fragment_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_global_context_index_input_loads_graph_for_pipeline_sentinel(
    monkeypatch,
):
    graph = CogneeGraph()
    child = _text_summary_graph_node()
    graph.add_node(child)
    get_fragment_mock = AsyncMock(return_value=graph)
    monkeypatch.setattr(load_module, "get_context_index_memory_fragment", get_fragment_mock)

    result = await load_module.extract_global_context_index_input([{}])

    assert [summary.id for summary in result.text_summaries] == [child.id]
    get_fragment_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_global_context_index_input_rejects_direct_input_data(monkeypatch):
    get_fragment_mock = AsyncMock(side_effect=AssertionError("should not load graph"))
    monkeypatch.setattr(load_module, "get_context_index_memory_fragment", get_fragment_mock)

    with pytest.raises(TypeError, match="expected None"):
        await load_module.extract_global_context_index_input(
            GlobalContextIndexInput(text_summaries=[], buckets=[])
        )

    get_fragment_mock.assert_not_awaited()


def _graph_input(
    entities_by_summary_id: dict[str, set[str]],
    idf_weights: dict[str, float],
) -> tuple[dict[str, set[str]], dict[str, float]]:
    return entities_by_summary_id, idf_weights


def test_extract_context_index_input_uses_summary_edges_for_assignment():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    bucket = _global_context_summary_graph_node(dataset_id)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(bucket)
    _add_summarized_in_edge(graph, child, bucket)

    summary_input = extract_context_index_input(graph, dataset_id)

    assert [summary.id for summary in summary_input.text_summaries] == [child.id]
    assert len(summary_input.buckets) == 1
    assert summary_input.buckets[0].id == bucket.id
    assert summary_input.buckets[0].child_ids == {child.id}
    assert summary_input.text_summaries[0].global_context_bucket_id == bucket.id


def test_extract_context_index_input_ignores_text_summary_bucket_marker():
    dataset_id = str(uuid4())
    bucket_id = str(uuid4())
    child = _text_summary_graph_node(bucket_id=bucket_id)
    bucket = _global_context_summary_graph_node(dataset_id, node_id=bucket_id)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(bucket)

    summary_input = extract_context_index_input(graph, dataset_id)

    assert summary_input.text_summaries[0].global_context_bucket_id is None
    assert summary_input.buckets[0].child_ids == set()


def test_extract_context_index_input_ignores_text_summary_to_root_edge():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    root = _global_context_summary_graph_node(dataset_id, is_root=True)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(root)
    _add_summarized_in_edge(graph, child, root)

    summary_input = extract_context_index_input(graph, dataset_id)

    assert summary_input.root is not None
    assert summary_input.root.child_ids == set()
    assert summary_input.text_summaries[0].global_context_bucket_id is None


def test_extract_context_index_input_ignores_stale_edge_to_missing_parent():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    missing_parent = _global_context_summary_graph_node(dataset_id)
    graph = CogneeGraph()
    graph.add_node(child)
    _add_summarized_in_edge(graph, child, missing_parent)

    summary_input = extract_context_index_input(graph, dataset_id)

    assert summary_input.text_summaries[0].global_context_bucket_id is None


def test_extract_context_index_input_ignores_bucket_marker_without_edge():
    dataset_id = str(uuid4())
    parent = _global_context_summary_graph_node(dataset_id, level=1)
    child = _global_context_summary_graph_node(dataset_id, level=0, bucket_id=parent.id)
    graph = CogneeGraph()
    graph.add_node(parent)
    graph.add_node(child)

    summary_input = extract_context_index_input(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id is None
    assert buckets_by_id[parent.id].child_ids == set()


def test_extract_context_index_input_uses_adjacent_bucket_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    parent = _global_context_summary_graph_node(dataset_id, level=1)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(parent)
    _add_summarized_in_edge(graph, child, parent)

    summary_input = extract_context_index_input(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id == parent.id
    assert buckets_by_id[parent.id].child_ids == {child.id}


def test_extract_context_index_input_uses_bucket_to_root_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    root = _global_context_summary_graph_node(dataset_id, is_root=True)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(root)
    _add_summarized_in_edge(graph, child, root)

    summary_input = extract_context_index_input(graph, dataset_id)

    assert summary_input.root is not None
    assert summary_input.buckets[0].global_context_bucket_id == root.id
    assert summary_input.root.child_ids == {child.id}


def test_extract_context_index_input_ignores_non_adjacent_bucket_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    parent = _global_context_summary_graph_node(dataset_id, level=2)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(parent)
    _add_summarized_in_edge(graph, child, parent)

    summary_input = extract_context_index_input(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id is None
    assert buckets_by_id[parent.id].child_ids == set()


def test_extract_context_index_input_filters_buckets_by_dataset_id():
    dataset_id = str(uuid4())
    dataset_name = "dataset-name"
    child = Node(str(uuid4()), {"type": "TextSummary", "text": "Chunk summary"})
    matching_bucket = Node(
        str(uuid4()),
        {
            "type": "GlobalContextSummary",
            "text": "Matching bucket",
            "dataset_id": dataset_id,
            "level": 0,
            "is_root": False,
        },
    )
    other_bucket = Node(
        str(uuid4()),
        {
            "type": "GlobalContextSummary",
            "text": "Other bucket",
            "dataset_id": str(uuid4()),
            "level": 0,
            "is_root": False,
        },
    )
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(matching_bucket)
    graph.add_node(other_bucket)

    summary_input = extract_context_index_input(graph, dataset_id)
    summary_input_from_name = extract_context_index_input(graph, dataset_name)

    assert [bucket.id for bucket in summary_input.buckets] == [matching_bucket.id]
    assert summary_input_from_name.buckets == []


def test_extract_context_index_input_rehydrates_graph_bucket_entity_state():
    dataset_id = str(uuid4())
    missing_state = _global_context_summary_graph_node(dataset_id, node_id="missing-state")
    null_state = _global_context_summary_graph_node(
        dataset_id,
        node_id="null-state",
        graph_bucket_entity_ids=None,
        include_graph_bucket_entity_ids=True,
    )
    empty_state = _global_context_summary_graph_node(
        dataset_id,
        node_id="empty-state",
        graph_bucket_entity_ids=[],
        include_graph_bucket_entity_ids=True,
    )
    populated_state = _global_context_summary_graph_node(
        dataset_id,
        node_id="populated-state",
        graph_bucket_entity_ids=["entity-b", "entity-a"],
        include_graph_bucket_entity_ids=True,
    )
    upper_level = _global_context_summary_graph_node(
        dataset_id,
        node_id="upper-level",
        level=1,
        graph_bucket_entity_ids=["ignored"],
        include_graph_bucket_entity_ids=True,
    )
    graph = CogneeGraph()
    for node in [missing_state, null_state, empty_state, populated_state, upper_level]:
        graph.add_node(node)

    summary_input = extract_context_index_input(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id["missing-state"].graph_bucket_entity_ids is None
    assert buckets_by_id["null-state"].graph_bucket_entity_ids is None
    assert buckets_by_id["empty-state"].graph_bucket_entity_ids == set()
    assert buckets_by_id["populated-state"].graph_bucket_entity_ids == {"entity-a", "entity-b"}
    assert buckets_by_id["upper-level"].graph_bucket_entity_ids is None


@pytest.mark.asyncio
async def test_build_bucket_summary_datapoint_persists_sorted_level_zero_graph_entity_state(
    monkeypatch,
):
    child_ids_seen = []

    async def summarize_bucket(children):
        child_ids_seen.extend(child.id for child in children)
        return "bucket summary"

    monkeypatch.setattr(
        summary_generation_module,
        "generate_bucket_summary",
        summarize_bucket,
    )
    bucket = SummaryNode(
        id=str(uuid4()),
        text="",
        type="GlobalContextSummary",
        level=0,
        child_ids={"summary-b", "summary-a"},
        graph_bucket_entity_ids={"entity-b", "entity-a"},
    )
    children_by_id = {
        "summary-a": SummaryNode(id="summary-a", text="A", type="TextSummary"),
        "summary-b": SummaryNode(id="summary-b", text="B", type="TextSummary"),
    }

    datapoint = await summary_generation_module.build_bucket_summary_datapoint(
        bucket,
        children_by_id,
        dataset_id=str(uuid4()),
    )

    assert child_ids_seen == ["summary-a", "summary-b"]
    assert datapoint.graph_bucket_entity_ids == ["entity-a", "entity-b"]


@pytest.mark.asyncio
async def test_build_bucket_summary_datapoint_leaves_non_level_zero_graph_state_unset(
    monkeypatch,
):
    async def summarize_bucket(children):
        return "bucket summary"

    monkeypatch.setattr(
        summary_generation_module,
        "generate_bucket_summary",
        summarize_bucket,
    )
    bucket = SummaryNode(
        id=str(uuid4()),
        text="",
        type="GlobalContextSummary",
        level=1,
        graph_bucket_entity_ids={"entity-a"},
    )

    datapoint = await summary_generation_module.build_bucket_summary_datapoint(
        bucket,
        children_by_id={},
        dataset_id=str(uuid4()),
    )

    assert datapoint.graph_bucket_entity_ids is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_config",
    [
        {"max_bucket_size": 1},
        {"max_bucket_size": 0},
        {"max_bucket_size": True},
        {"max_bucket_size": 1.5},
        {"placement_distance_threshold": -1},
        {"placement_distance_threshold": True},
        {"placement_distance_threshold": "0.5"},
        {"placement_distance_threshold": float("inf")},
        {"placement_distance_threshold": float("nan")},
        {"bucketing_strategy": "unknown"},
        {"min_overlap": -0.1},
        {"min_overlap": 1.1},
        {"min_overlap": True},
        {"min_overlap": "0.1"},
        {"min_overlap": float("inf")},
        {"min_overlap": float("nan")},
    ],
)
async def test_update_global_context_index_rejects_invalid_config_before_io(
    monkeypatch,
    invalid_config,
):
    load_summary_input_mock = AsyncMock()
    get_unified_engine_mock = AsyncMock()
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_context_index_input_from_graph",
        load_summary_input_mock,
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        get_unified_engine_mock,
    )

    with pytest.raises(ValueError):
        await update_global_context_index(
            GlobalContextIndexInput(text_summaries=[], buckets=[]),
            **invalid_config,
        )

    load_summary_input_mock.assert_not_awaited()
    get_unified_engine_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_rejects_graph_without_dataset_before_io(monkeypatch):
    load_summary_input_mock = AsyncMock()
    get_unified_engine_mock = AsyncMock()
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_context_index_input_from_graph",
        load_summary_input_mock,
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        get_unified_engine_mock,
    )

    with pytest.raises(ValueError, match="requires a dataset context"):
        await update_global_context_index(
            bucketing_strategy="graph",
            rebuild=False,
        )

    load_summary_input_mock.assert_not_awaited()
    get_unified_engine_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_first_build_creates_one_bucket_per_summary(monkeypatch):
    summaries = [_summary_node("alpha"), _summary_node("beta"), _summary_node("gamma")]
    summary_input = GlobalContextIndexInput(text_summaries=summaries, buckets=[])
    dataset_id = uuid4()
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(search=AsyncMock(return_value=[]), delete_data_points=AsyncMock()),
    )
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return " / ".join(child.text for child in children)

    async def summarize_root(children):
        return "root: " + " / ".join(child.text for child in children)

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset_id, name="dataset-name"))
    result = await update_global_context_index(
        summary_input,
        max_bucket_size=10,
        ctx=ctx,
    )

    non_root = [dp for dp in result if not dp.is_root]
    roots = [dp for dp in result if dp.is_root]
    assert len(non_root) == 3
    assert len(roots) == 1
    assert {bucket.dataset_id for bucket in non_root} == {str(dataset_id)}
    assert "dataset-name" not in {bucket.dataset_id for bucket in result}
    assert all(isinstance(dp, GlobalContextSummary) for dp in result)
    edge_args = unified_engine.graph.add_edges.await_args.args[0]
    # 3 TextSummary -> bucket and 3 bucket -> root.
    assert len(edge_args) == 6
    assert {edge[2] for edge in edge_args} == {SUMMARIZED_IN}
    assert {edge[3]["relationship_name"] for edge in edge_args} == {SUMMARIZED_IN}
    assert {summary.global_context_bucket_id for summary in summaries} == {
        str(bucket.id) for bucket in non_root
    }
    _assert_add_data_points_used_context(add_data_points_mock, ctx)
    unified_engine.graph.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_persists_summaries_before_edges(monkeypatch):
    events = []
    summaries = [_summary_node("alpha"), _summary_node("beta"), _summary_node("gamma")]
    dataset_id = uuid4()
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(side_effect=lambda edges: events.append(("edges", len(edges)))),
        ),
        vector=SimpleNamespace(search=AsyncMock(return_value=[]), delete_data_points=AsyncMock()),
    )

    async def add_data_points(datapoints, ctx=None):
        events.append(("summaries", len(datapoints)))

    async def summarize_bucket(children):
        return " / ".join(child.text for child in children)

    async def summarize_root(children):
        return "root: " + " / ".join(child.text for child in children)

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points,
    )

    await update_global_context_index(
        GlobalContextIndexInput(text_summaries=summaries, buckets=[]),
        max_bucket_size=10,
        ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
    )

    assert [event[0] for event in events] == ["summaries", "summaries", "edges"]
    assert events[-1][1] == 6


@pytest.mark.asyncio
async def test_update_global_context_index_builds_multi_level_index(monkeypatch):
    summaries = [_summary_node(f"summary-{index}") for index in range(5)]
    dataset_id = uuid4()
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(search=AsyncMock(return_value=[]), delete_data_points=AsyncMock()),
    )
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return "bucket(" + " | ".join(child.text for child in children) + ")"

    async def summarize_root(children):
        return "root(" + " | ".join(child.text for child in children) + ")"

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset_id))
    result = await update_global_context_index(
        GlobalContextIndexInput(text_summaries=summaries, buckets=[]),
        max_bucket_size=2,
        ctx=ctx,
    )

    roots = [datapoint for datapoint in result if datapoint.is_root]
    non_root = [datapoint for datapoint in result if not datapoint.is_root]
    upper_level = [datapoint for datapoint in non_root if datapoint.level > 0]

    assert len(roots) == 1
    assert upper_level
    assert roots[0].level == max(datapoint.level for datapoint in non_root) + 1
    assert all(datapoint.text.startswith("bucket(bucket(") for datapoint in upper_level)

    edge_args = unified_engine.graph.add_edges.await_args.args[0]
    upward_child_ids = {edge[0] for edge in edge_args}
    non_root_bucket_ids = {str(datapoint.id) for datapoint in non_root}
    assert non_root_bucket_ids <= upward_child_ids
    assert {edge[2] for edge in edge_args} == {SUMMARIZED_IN}
    _assert_add_data_points_used_context(add_data_points_mock, ctx)


@pytest.mark.asyncio
async def test_update_global_context_index_places_new_summary_into_existing_bucket(monkeypatch):
    bucket_id = str(uuid4())
    existing_child = _summary_node("storage uses postgres", bucket_id=bucket_id)
    new_child = _summary_node("postgres remains the storage choice")
    bucket = SummaryNode(
        id=bucket_id,
        text="storage bucket",
        type="GlobalContextSummary",
        level=0,
        dataset_id=str(uuid4()),
        child_ids={existing_child.id},
    )
    summary_input = GlobalContextIndexInput(
        text_summaries=[existing_child, new_child],
        buckets=[bucket],
    )
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(
            search=AsyncMock(return_value=[SimpleNamespace(id=existing_child.id, score=0.1)]),
            delete_data_points=AsyncMock(),
        ),
    )
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return " | ".join(sorted(child.text for child in children))

    async def summarize_root(children):
        return "root: " + " | ".join(sorted(child.text for child in children))

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    ctx = PipelineContext(dataset=SimpleNamespace(id=bucket.dataset_id))
    result = await update_global_context_index(
        summary_input,
        placement_distance_threshold=0.75,
        ctx=ctx,
    )

    non_root = [dp for dp in result if not dp.is_root]
    roots = [dp for dp in result if dp.is_root]
    assert len(non_root) == 1
    assert str(non_root[0].id) == bucket.id
    assert len(roots) == 1
    edge_args = unified_engine.graph.add_edges.await_args.args[0]
    edge_pairs = [(edge[0], edge[1]) for edge in edge_args]
    assert (new_child.id, bucket.id) in edge_pairs
    assert (bucket.id, str(roots[0].id)) in edge_pairs
    assert new_child.global_context_bucket_id == bucket.id
    _assert_add_data_points_used_context(add_data_points_mock, ctx)
    unified_engine.graph.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_creates_new_bucket_when_no_existing_bucket_matches(
    monkeypatch,
):
    bucket_id = str(uuid4())
    existing_child = _summary_node("existing topic", bucket_id=bucket_id)
    new_child_1 = _summary_node("new topic first")
    existing_bucket = SummaryNode(
        id=bucket_id,
        text="existing bucket",
        type="GlobalContextSummary",
        level=0,
        dataset_id=str(uuid4()),
        child_ids={existing_child.id},
    )

    async def vector_search(collection_name, query_text, limit=None, include_payload=False):
        return [SimpleNamespace(id=existing_bucket.id, score=0.9)]

    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(
            search=AsyncMock(side_effect=vector_search),
            delete_data_points=AsyncMock(),
        ),
    )
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return " / ".join(sorted(child.text for child in children))

    async def summarize_root(children):
        return "root: " + " / ".join(sorted(child.text for child in children))

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    ctx = PipelineContext(dataset=SimpleNamespace(id=existing_bucket.dataset_id))
    result = await update_global_context_index(
        GlobalContextIndexInput(
            text_summaries=[existing_child, new_child_1],
            buckets=[existing_bucket],
        ),
        max_bucket_size=10,
        placement_distance_threshold=0.5,
        ctx=ctx,
    )

    non_root = [dp for dp in result if not dp.is_root]
    assert len(non_root) == 1
    assert str(non_root[0].id) != existing_bucket.id
    edge_args = unified_engine.graph.add_edges.await_args.args[0]
    edge_pairs = [(edge[0], edge[1]) for edge in edge_args]
    assert (new_child_1.id, str(non_root[0].id)) in edge_pairs
    _assert_add_data_points_used_context(add_data_points_mock, ctx)
    unified_engine.graph.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_skips_marked_text_summaries(monkeypatch):
    bucket_id = str(uuid4())
    assigned_child = _summary_node("already assigned", bucket_id=bucket_id)
    unassigned_child = _summary_node("new unassigned")
    existing_bucket = SummaryNode(
        id=bucket_id,
        text="existing bucket",
        type="GlobalContextSummary",
        level=0,
        dataset_id=str(uuid4()),
        global_context_bucket_id=None,
        child_ids={assigned_child.id},
    )
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(
            search=AsyncMock(return_value=[SimpleNamespace(id=assigned_child.id, score=0.1)]),
            delete_data_points=AsyncMock(),
        ),
    )
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return " / ".join(sorted(child.text for child in children))

    async def summarize_root(children):
        return "root: " + " / ".join(sorted(child.text for child in children))

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    ctx = PipelineContext(dataset=SimpleNamespace(id=existing_bucket.dataset_id))
    result = await update_global_context_index(
        GlobalContextIndexInput(
            text_summaries=[assigned_child, unassigned_child],
            buckets=[existing_bucket],
        ),
        placement_distance_threshold=0.5,
        ctx=ctx,
    )

    non_root = [dp for dp in result if not dp.is_root]
    assert len(non_root) == 1
    unified_engine.vector.search.assert_awaited_once()
    assert unassigned_child.global_context_bucket_id == bucket_id
    _assert_add_data_points_used_context(add_data_points_mock, ctx)
    unified_engine.graph.add_nodes.assert_not_awaited()
    edge_args = unified_engine.graph.add_edges.await_args.args[0]
    edge_pairs = [(edge[0], edge[1]) for edge in edge_args]
    assert (unassigned_child.id, bucket_id) in edge_pairs


@pytest.mark.asyncio
async def test_update_global_context_index_rejects_vector_incremental_on_graph_buckets(
    monkeypatch,
):
    bucket_id = str(uuid4())
    assigned_child = _summary_node("already assigned", bucket_id=bucket_id)
    unassigned_child = _summary_node("new unassigned")
    graph_bucket = SummaryNode(
        id=bucket_id,
        text="graph bucket",
        type="GlobalContextSummary",
        level=0,
        dataset_id=str(uuid4()),
        child_ids={assigned_child.id},
        graph_bucket_entity_ids={"entity-a"},
    )
    unified_engine = _unified_engine()
    _stub_global_context_index_io(monkeypatch, unified_engine)

    with pytest.raises(ValueError, match="cannot extend graph-built"):
        await update_global_context_index(
            GlobalContextIndexInput(
                text_summaries=[assigned_child, unassigned_child],
                buckets=[graph_bucket],
            ),
            bucketing_strategy="vector",
            rebuild=False,
            ctx=PipelineContext(dataset=SimpleNamespace(id=graph_bucket.dataset_id)),
        )

    unified_engine.vector.search.assert_not_awaited()
    unified_engine.graph.add_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_rebuild_empty_input_deletes_old_index(
    monkeypatch,
):
    bucket = SummaryNode(
        id=str(uuid4()),
        text="old bucket",
        type="GlobalContextSummary",
        level=0,
        dataset_id=str(uuid4()),
    )
    root = SummaryNode(
        id=str(uuid4()),
        text="old root",
        type="GlobalContextSummary",
        level=1,
        is_root=True,
        dataset_id=bucket.dataset_id,
    )
    empty_summary = _summary_node("")
    unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(search=AsyncMock(), delete_data_points=AsyncMock()),
    )
    run_sweep_mock = AsyncMock()
    add_data_points_mock = AsyncMock()
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "build_context_index",
        run_sweep_mock,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )

    result = await update_global_context_index(
        GlobalContextIndexInput(
            text_summaries=[empty_summary],
            buckets=[bucket],
            root=root,
        ),
        rebuild=True,
        ctx=PipelineContext(dataset=SimpleNamespace(id=bucket.dataset_id)),
    )

    assert result == []
    unified_engine.graph.delete_nodes.assert_awaited_once_with([bucket.id, root.id])
    unified_engine.vector.delete_data_points.assert_awaited_once()
    run_sweep_mock.assert_not_awaited()
    unified_engine.vector.search.assert_not_awaited()
    add_data_points_mock.assert_not_awaited()
    unified_engine.graph.add_nodes.assert_not_awaited()
    unified_engine.graph.add_edges.assert_not_awaited()


def _unified_engine(vector_search=None):
    return SimpleNamespace(
        graph=SimpleNamespace(
            delete_nodes=AsyncMock(),
            add_nodes=AsyncMock(),
            add_edges=AsyncMock(),
        ),
        vector=SimpleNamespace(
            search=AsyncMock(side_effect=vector_search)
            if vector_search is not None
            else AsyncMock(return_value=[]),
            delete_data_points=AsyncMock(),
        ),
    )


def _stub_global_context_index_io(monkeypatch, unified_engine):
    add_data_points_mock = AsyncMock()

    async def summarize_bucket(children):
        return " / ".join(child.text for child in children)

    async def summarize_root(children):
        return "root: " + " / ".join(child.text for child in children)

    monkeypatch.setattr(
        update_global_context_index_module,
        "get_unified_engine",
        AsyncMock(return_value=unified_engine),
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summarize.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persist.add_data_points",
        add_data_points_mock,
    )
    return add_data_points_mock


@pytest.mark.asyncio
async def test_update_global_context_index_graph_rebuild_uses_provider_entities(monkeypatch):
    summaries = [
        _summary_node("alice roadmap"),
        _summary_node("alice launch"),
        _summary_node("bob migration"),
    ]
    foreign_summary = _summary_node("foreign dataset summary")
    dataset_id = uuid4()
    unified_engine = _unified_engine()
    add_data_points_mock = _stub_global_context_index_io(monkeypatch, unified_engine)
    dataset_summary_ids_mock = AsyncMock(return_value=[summary.id for summary in summaries])
    provider_mock = AsyncMock(
        return_value=_graph_input(
            {
                summaries[0].id: {"alice"},
                summaries[1].id: {"alice"},
                summaries[2].id: {"bob"},
            },
            {"alice": 1.0, "bob": 1.0},
        )
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        dataset_summary_ids_mock,
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        provider_mock,
    )
    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset_id))

    result = await update_global_context_index(
        GlobalContextIndexInput(
            text_summaries=[*summaries, foreign_summary],
            buckets=[],
            entities_by_summary_id={
                summary.id: {"ignored"} for summary in [*summaries, foreign_summary]
            },
        ),
        rebuild=True,
        bucketing_strategy="graph",
        max_bucket_size=10,
        ctx=ctx,
    )

    dataset_summary_ids_mock.assert_awaited_once_with(str(dataset_id))
    provider_mock.assert_awaited_once_with(
        str(dataset_id),
        sorted(summary.id for summary in summaries),
    )
    unified_engine.vector.search.assert_not_awaited()
    assert summaries[0].global_context_bucket_id == summaries[1].global_context_bucket_id
    assert summaries[2].global_context_bucket_id != summaries[0].global_context_bucket_id
    assert foreign_summary.global_context_bucket_id is None

    level_zero = [datapoint for datapoint in result if datapoint.level == 0]
    assert sorted(datapoint.graph_bucket_entity_ids for datapoint in level_zero) == [
        ["alice"],
        ["bob"],
    ]
    _assert_add_data_points_used_context(add_data_points_mock, ctx)


@pytest.mark.asyncio
async def test_update_global_context_index_graph_rebuild_rejects_missing_dataset_summary(
    monkeypatch,
):
    loaded_summary = _summary_node("loaded")
    missing_summary_id = str(uuid4())
    dataset_id = uuid4()
    unified_engine = _unified_engine()
    _stub_global_context_index_io(monkeypatch, unified_engine)
    provider_mock = AsyncMock()
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        AsyncMock(return_value=[loaded_summary.id, missing_summary_id]),
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        provider_mock,
    )

    with pytest.raises(ValueError, match="could not load graph TextSummary"):
        await update_global_context_index(
            GlobalContextIndexInput(text_summaries=[loaded_summary], buckets=[]),
            rebuild=True,
            bucketing_strategy="graph",
            ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
        )

    provider_mock.assert_not_awaited()
    unified_engine.graph.delete_nodes.assert_not_awaited()
    unified_engine.graph.add_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_graph_incremental_no_new_summaries_skips_graph_input(
    monkeypatch,
):
    bucket_id = str(uuid4())
    assigned_child = _summary_node("already assigned", bucket_id=bucket_id)
    graph_bucket = SummaryNode(
        id=bucket_id,
        text="alice bucket",
        type="GlobalContextSummary",
        level=0,
        child_ids={assigned_child.id},
        graph_bucket_entity_ids={"alice"},
    )
    dataset_id = uuid4()
    unified_engine = _unified_engine()
    _stub_global_context_index_io(monkeypatch, unified_engine)
    provider_mock = AsyncMock()
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        AsyncMock(return_value=[assigned_child.id]),
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        provider_mock,
    )

    result = await update_global_context_index(
        GlobalContextIndexInput(text_summaries=[assigned_child], buckets=[graph_bucket]),
        bucketing_strategy="graph",
        rebuild=False,
        ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
    )

    assert result == []
    provider_mock.assert_not_awaited()
    unified_engine.vector.search.assert_not_awaited()
    unified_engine.graph.add_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_global_context_index_graph_strategy_uses_vector_above_level_zero(
    monkeypatch,
):
    summaries = [_summary_node(f"summary-{index}") for index in range(5)]
    dataset_id = uuid4()

    async def vector_search(collection_name, query_text, limit=None, include_payload=False):
        assert collection_name == "GlobalContextSummary_text"
        return []

    unified_engine = _unified_engine(vector_search=vector_search)
    _stub_global_context_index_io(monkeypatch, unified_engine)
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        AsyncMock(return_value=[summary.id for summary in summaries]),
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        AsyncMock(
            return_value=_graph_input(
                {summary.id: {f"entity-{index}"} for index, summary in enumerate(summaries)},
                {f"entity-{index}": 1.0 for index in range(len(summaries))},
            )
        ),
    )

    result = await update_global_context_index(
        GlobalContextIndexInput(text_summaries=summaries, buckets=[]),
        rebuild=True,
        bucketing_strategy="graph",
        max_bucket_size=2,
        ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
    )

    assert any(datapoint.level > 0 and not datapoint.is_root for datapoint in result)
    assert unified_engine.vector.search.await_count > 0


@pytest.mark.asyncio
async def test_update_global_context_index_graph_incremental_extends_graph_bucket(monkeypatch):
    bucket_id = str(uuid4())
    assigned_child = _summary_node("alice roadmap", bucket_id=bucket_id)
    new_child = _summary_node("alice launch")
    graph_bucket = SummaryNode(
        id=bucket_id,
        text="alice bucket",
        type="GlobalContextSummary",
        level=0,
        child_ids={assigned_child.id},
        graph_bucket_entity_ids={"alice"},
    )
    dataset_id = uuid4()
    unified_engine = _unified_engine()
    _stub_global_context_index_io(monkeypatch, unified_engine)
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        AsyncMock(return_value=[assigned_child.id, new_child.id]),
    )
    provider_mock = AsyncMock(
        return_value=_graph_input(
            {
                assigned_child.id: {"alice"},
                new_child.id: {"alice"},
            },
            {"alice": 1.0},
        )
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        provider_mock,
    )

    result = await update_global_context_index(
        GlobalContextIndexInput(
            text_summaries=[assigned_child, new_child],
            buckets=[graph_bucket],
        ),
        bucketing_strategy="graph",
        rebuild=False,
        max_bucket_size=3,
        ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
    )

    provider_summary_ids = provider_mock.await_args.args[1]
    assert set(provider_summary_ids) == {assigned_child.id, new_child.id}
    assert graph_bucket.child_ids == {assigned_child.id, new_child.id}
    assert new_child.global_context_bucket_id == bucket_id
    unified_engine.graph.delete_nodes.assert_not_awaited()
    unified_engine.vector.search.assert_not_awaited()

    level_zero = [datapoint for datapoint in result if datapoint.level == 0]
    assert len(level_zero) == 1
    assert str(level_zero[0].id) == bucket_id
    assert level_zero[0].graph_bucket_entity_ids == ["alice"]

    edge_pairs = [(edge[0], edge[1]) for edge in unified_engine.graph.add_edges.await_args.args[0]]
    assert (new_child.id, bucket_id) in edge_pairs


@pytest.mark.asyncio
async def test_update_global_context_index_graph_rebuild_rejects_missing_made_from_before_delete(
    monkeypatch,
):
    summary = _summary_node("summary")
    dataset_id = uuid4()
    existing_bucket = SummaryNode(
        id=str(uuid4()),
        text="old bucket",
        type="GlobalContextSummary",
        level=0,
        child_ids={summary.id},
        graph_bucket_entity_ids={"old-entity"},
    )
    existing_root = SummaryNode(
        id=str(uuid4()),
        text="old root",
        type="GlobalContextSummary",
        level=1,
        is_root=True,
    )
    unified_engine = _unified_engine()
    _stub_global_context_index_io(monkeypatch, unified_engine)
    monkeypatch.setattr(
        update_global_context_index_module,
        "get_dataset_text_summary_ids",
        AsyncMock(return_value=[summary.id]),
    )
    monkeypatch.setattr(
        update_global_context_index_module,
        "load_graph_bucketing_inputs",
        AsyncMock(
            side_effect=ValueError(
                'bucketing_strategy="graph" requires every TextSummary to have a made_from '
                f"chunk edge. Missing made_from for 1 summary id(s): {summary.id}"
            )
        ),
    )

    with pytest.raises(ValueError, match="made_from"):
        await update_global_context_index(
            GlobalContextIndexInput(
                text_summaries=[summary],
                buckets=[existing_bucket],
                root=existing_root,
            ),
            rebuild=True,
            bucketing_strategy="graph",
            ctx=PipelineContext(dataset=SimpleNamespace(id=dataset_id)),
        )

    unified_engine.graph.delete_nodes.assert_not_awaited()
    unified_engine.vector.delete_data_points.assert_not_awaited()
    unified_engine.graph.add_edges.assert_not_awaited()
