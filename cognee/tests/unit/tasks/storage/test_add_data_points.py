import pytest
from unittest.mock import AsyncMock, patch
import sys

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Triplet
from cognee.tasks.storage.add_data_points import (
    add_data_points,
    InvalidDataPointsInAddDataPointsError,
    _extract_embeddable_text_from_datapoint,
    _create_triplets_from_graph,
)

adp_module = sys.modules["cognee.tasks.storage.add_data_points"]


class SimplePoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_input", [None, ["not_datapoint"]])
async def test_add_data_points_validates_inputs(bad_input):
    with pytest.raises(InvalidDataPointsInAddDataPointsError):
        await add_data_points(bad_input)


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_graph_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_indexes_nodes_and_edges(
    mock_get_graph, mock_dedup, mock_get_engine, mock_index_nodes, mock_index_edges
):
    dp1 = SimplePoint(text="first")
    dp2 = SimplePoint(text="second")

    edge1 = (str(dp1.id), str(dp2.id), "related_to", {"edge_text": "connects"})
    custom_edges = [(str(dp2.id), str(dp1.id), "custom_edge", {})]

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    graph_engine = AsyncMock()
    mock_get_engine.return_value = graph_engine

    result = await add_data_points([dp1, dp2], custom_edges=custom_edges)

    assert result == [dp1, dp2]
    graph_engine.add_nodes.assert_awaited_once()
    mock_index_nodes.assert_awaited_once()
    assert graph_engine.add_edges.await_count == 2
    assert edge1 in graph_engine.add_edges.await_args_list[0].args[0]
    assert graph_engine.add_edges.await_args_list[1].args[0] == custom_edges
    assert mock_index_edges.await_count == 2


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_graph_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_indexes_triplets_when_enabled(
    mock_get_graph, mock_dedup, mock_get_engine, mock_index_nodes, mock_index_edges
):
    dp1 = SimplePoint(text="source")
    dp2 = SimplePoint(text="target")

    edge1 = (str(dp1.id), str(dp2.id), "relates", {"edge_text": "describes"})

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    graph_engine = AsyncMock()
    mock_get_engine.return_value = graph_engine

    await add_data_points([dp1, dp2], embed_triplets=True)

    assert mock_index_nodes.await_count == 2
    nodes_arg = mock_index_nodes.await_args_list[0].args[0]
    triplets_arg = mock_index_nodes.await_args_list[1].args[0]
    assert nodes_arg == [dp1, dp2]
    assert len(triplets_arg) == 1
    assert isinstance(triplets_arg[0], Triplet)
    mock_index_edges.assert_awaited_once()


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_graph_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_empty_list(
    mock_get_graph, mock_dedup, mock_get_engine, mock_index_nodes, mock_index_edges
):
    mock_dedup.side_effect = lambda n, e: (n, e)
    graph_engine = AsyncMock()
    mock_get_engine.return_value = graph_engine

    result = await add_data_points([])

    assert result == []
    mock_get_graph.assert_not_called()
    graph_engine.add_nodes.assert_awaited_once_with([])


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_graph_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_single_datapoint(
    mock_get_graph, mock_dedup, mock_get_engine, mock_index_nodes, mock_index_edges
):
    dp = SimplePoint(text="single")
    mock_get_graph.side_effect = [([dp], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    graph_engine = AsyncMock()
    mock_get_engine.return_value = graph_engine

    result = await add_data_points([dp])

    assert result == [dp]
    mock_get_graph.assert_called_once()
    mock_index_nodes.assert_awaited_once()


def test_extract_embeddable_text_from_datapoint():
    dp = SimplePoint(text="hello world")
    text = _extract_embeddable_text_from_datapoint(dp)
    assert text == "hello world"


def test_extract_embeddable_text_with_multiple_fields():
    class MultiField(DataPoint):
        title: str
        description: str
        metadata: dict = {"index_fields": ["title", "description"]}

    dp = MultiField(title="Test", description="Description")
    text = _extract_embeddable_text_from_datapoint(dp)
    assert text == "Test Description"


def test_extract_embeddable_text_with_no_index_fields():
    class NoIndex(DataPoint):
        text: str
        metadata: dict = {"index_fields": []}

    dp = NoIndex(text="ignored")
    text = _extract_embeddable_text_from_datapoint(dp)
    assert text == ""


def test_create_triplets_from_graph():
    dp1 = SimplePoint(text="source node")
    dp2 = SimplePoint(text="target node")
    edge = (str(dp1.id), str(dp2.id), "connects_to", {"edge_text": "links"})

    triplets = _create_triplets_from_graph([dp1, dp2], [edge])

    assert len(triplets) == 1
    assert isinstance(triplets[0], Triplet)
    assert triplets[0].from_node_id == str(dp1.id)
    assert triplets[0].to_node_id == str(dp2.id)
    assert "source node" in triplets[0].text
    assert "target node" in triplets[0].text


def test_extract_embeddable_text_with_none_datapoint():
    text = _extract_embeddable_text_from_datapoint(None)
    assert text == ""


def test_extract_embeddable_text_without_metadata():
    class NoMetadata(DataPoint):
        text: str

    dp = NoMetadata(text="test")
    delattr(dp, "metadata")
    text = _extract_embeddable_text_from_datapoint(dp)
    assert text == ""


def test_extract_embeddable_text_with_whitespace_only():
    class WhitespaceField(DataPoint):
        text: str
        metadata: dict = {"index_fields": ["text"]}

    dp = WhitespaceField(text="   ")
    text = _extract_embeddable_text_from_datapoint(dp)
    assert text == ""


def test_create_triplets_skips_short_edge_tuples():
    dp = SimplePoint(text="node")
    incomplete_edge = (str(dp.id), str(dp.id))

    triplets = _create_triplets_from_graph([dp], [incomplete_edge])

    assert len(triplets) == 0


def test_create_triplets_skips_missing_source_node():
    dp1 = SimplePoint(text="target")
    edge = ("missing_id", str(dp1.id), "relates", {})

    triplets = _create_triplets_from_graph([dp1], [edge])

    assert len(triplets) == 0


def test_create_triplets_skips_missing_target_node():
    dp1 = SimplePoint(text="source")
    edge = (str(dp1.id), "missing_id", "relates", {})

    triplets = _create_triplets_from_graph([dp1], [edge])

    assert len(triplets) == 0


def test_create_triplets_skips_none_relationship():
    dp1 = SimplePoint(text="source")
    dp2 = SimplePoint(text="target")
    edge = (str(dp1.id), str(dp2.id), None, {})

    triplets = _create_triplets_from_graph([dp1, dp2], [edge])

    assert len(triplets) == 0


def test_create_triplets_uses_relationship_name_when_no_edge_text():
    dp1 = SimplePoint(text="source")
    dp2 = SimplePoint(text="target")
    edge = (str(dp1.id), str(dp2.id), "connects_to", {})

    triplets = _create_triplets_from_graph([dp1, dp2], [edge])

    assert len(triplets) == 1
    assert "connects_to" in triplets[0].text


def test_create_triplets_prevents_duplicates():
    dp1 = SimplePoint(text="source")
    dp2 = SimplePoint(text="target")
    edge = (str(dp1.id), str(dp2.id), "relates", {"edge_text": "links"})

    triplets = _create_triplets_from_graph([dp1, dp2], [edge, edge])

    assert len(triplets) == 1


def test_create_triplets_skips_nodes_without_id():
    class NodeNoId:
        pass

    dp = SimplePoint(text="valid")
    node_no_id = NodeNoId()
    edge = (str(dp.id), "some_id", "relates", {})

    triplets = _create_triplets_from_graph([dp, node_no_id], [edge])

    assert len(triplets) == 0


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_graph_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_empty_custom_edges(
    mock_get_graph, mock_dedup, mock_get_engine, mock_index_nodes, mock_index_edges
):
    dp = SimplePoint(text="test")
    mock_get_graph.side_effect = [([dp], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    graph_engine = AsyncMock()
    mock_get_engine.return_value = graph_engine

    result = await add_data_points([dp], custom_edges=[])

    assert result == [dp]
    assert graph_engine.add_edges.await_count == 1
