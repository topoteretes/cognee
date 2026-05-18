from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.memify.global_context_index.constants import SUMMARIZED_IN
from cognee.tasks.memify.global_context_index.graph_input import (
    extract_context_index_input_from_graph,
)
from cognee.tasks.memify.global_context_index.models import GlobalContextIndexInput, SummaryNode
from cognee.tasks.memify.global_context_index.update_global_context_index import (
    update_global_context_index,
)
from cognee.tasks.summarization.models import GlobalContextSummary

update_global_context_index_module = import_module(
    "cognee.tasks.memify.global_context_index.update_global_context_index"
)


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


def test_extract_context_index_input_from_graph_uses_summary_edges_for_assignment():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    bucket = _global_context_summary_graph_node(dataset_id)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(bucket)
    _add_summarized_in_edge(graph, child, bucket)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    assert [summary.id for summary in summary_input.text_summaries] == [child.id]
    assert len(summary_input.buckets) == 1
    assert summary_input.buckets[0].id == bucket.id
    assert summary_input.buckets[0].child_ids == {child.id}
    assert summary_input.text_summaries[0].global_context_bucket_id == bucket.id


def test_extract_context_index_input_from_graph_ignores_text_summary_bucket_marker():
    dataset_id = str(uuid4())
    bucket_id = str(uuid4())
    child = _text_summary_graph_node(bucket_id=bucket_id)
    bucket = _global_context_summary_graph_node(dataset_id, node_id=bucket_id)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(bucket)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    assert summary_input.text_summaries[0].global_context_bucket_id is None
    assert summary_input.buckets[0].child_ids == set()


def test_extract_context_index_input_from_graph_ignores_text_summary_to_root_edge():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    root = _global_context_summary_graph_node(dataset_id, is_root=True)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(root)
    _add_summarized_in_edge(graph, child, root)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    assert summary_input.root is not None
    assert summary_input.root.child_ids == set()
    assert summary_input.text_summaries[0].global_context_bucket_id is None


def test_extract_context_index_input_from_graph_ignores_stale_edge_to_missing_parent():
    dataset_id = str(uuid4())
    child = _text_summary_graph_node()
    missing_parent = _global_context_summary_graph_node(dataset_id)
    graph = CogneeGraph()
    graph.add_node(child)
    _add_summarized_in_edge(graph, child, missing_parent)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    assert summary_input.text_summaries[0].global_context_bucket_id is None


def test_extract_context_index_input_from_graph_ignores_bucket_marker_without_edge():
    dataset_id = str(uuid4())
    parent = _global_context_summary_graph_node(dataset_id, level=1)
    child = _global_context_summary_graph_node(dataset_id, level=0, bucket_id=parent.id)
    graph = CogneeGraph()
    graph.add_node(parent)
    graph.add_node(child)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id is None
    assert buckets_by_id[parent.id].child_ids == set()


def test_extract_context_index_input_from_graph_uses_adjacent_bucket_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    parent = _global_context_summary_graph_node(dataset_id, level=1)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(parent)
    _add_summarized_in_edge(graph, child, parent)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id == parent.id
    assert buckets_by_id[parent.id].child_ids == {child.id}


def test_extract_context_index_input_from_graph_uses_bucket_to_root_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    root = _global_context_summary_graph_node(dataset_id, is_root=True)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(root)
    _add_summarized_in_edge(graph, child, root)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    assert summary_input.root is not None
    assert summary_input.buckets[0].global_context_bucket_id == root.id
    assert summary_input.root.child_ids == {child.id}


def test_extract_context_index_input_from_graph_ignores_non_adjacent_bucket_edge():
    dataset_id = str(uuid4())
    child = _global_context_summary_graph_node(dataset_id, level=0)
    parent = _global_context_summary_graph_node(dataset_id, level=2)
    graph = CogneeGraph()
    graph.add_node(child)
    graph.add_node(parent)
    _add_summarized_in_edge(graph, child, parent)

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)

    buckets_by_id = {bucket.id: bucket for bucket in summary_input.buckets}
    assert buckets_by_id[child.id].global_context_bucket_id is None
    assert buckets_by_id[parent.id].child_ids == set()


def test_extract_context_index_input_from_graph_filters_buckets_by_dataset_id():
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

    summary_input = extract_context_index_input_from_graph(graph, dataset_id)
    summary_input_from_name = extract_context_index_input_from_graph(graph, dataset_name)

    assert [bucket.id for bucket in summary_input.buckets] == [matching_bucket.id]
    assert summary_input_from_name.buckets == []


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
        "load_context_index_input",
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
        "cognee.tasks.memify.global_context_index.summary_generation.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summary_generation.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
        "cognee.tasks.memify.global_context_index.summary_generation.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summary_generation.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
        "cognee.tasks.memify.global_context_index.summary_generation.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summary_generation.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
        "cognee.tasks.memify.global_context_index.summary_generation.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summary_generation.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
        "cognee.tasks.memify.global_context_index.summary_generation.generate_bucket_summary",
        summarize_bucket,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.summary_generation.generate_global_context_summary",
        summarize_root,
    )
    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
        "cognee.tasks.memify.global_context_index.persistence.add_data_points",
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
