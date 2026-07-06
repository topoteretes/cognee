import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from types import SimpleNamespace
from uuid import uuid4

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.models import Triplet
from cognee.modules.graph.utils import ensure_default_edge_properties
from cognee.modules.pipelines.models import PipelineContext
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


class NamedPoint(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class TitledPoint(DataPoint):
    title: str
    metadata: dict = {"index_fields": ["title"]}


class IndexedNonNamePoint(DataPoint):
    """`name` is present, but `index_fields` points to a different field."""

    name: str
    handle: str
    metadata: dict = {"index_fields": ["handle"]}


class OnlyNamePoint(DataPoint):
    """Has `name` but no `index_fields` declared."""

    name: str


class UnlabelablePoint(DataPoint):
    """Has neither `name` nor `index_fields` — the misuse case."""

    payload: str


def _make_unified_mock():
    """Create a mock UnifiedStoreEngine with graph and vector properties."""
    graph_engine = AsyncMock()
    vector_engine = MagicMock()
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = vector_engine
    unified.has_capability = MagicMock(return_value=False)
    return unified, graph_engine, vector_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_input", [None, ["not_datapoint"]])
async def test_add_data_points_validates_inputs(bad_input):
    with pytest.raises(InvalidDataPointsInAddDataPointsError):
        await add_data_points(bad_input)


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_indexes_nodes_and_edges(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    dp1 = SimplePoint(text="first")
    dp2 = SimplePoint(text="second")

    edge1 = (str(dp1.id), str(dp2.id), "related_to", {"edge_text": "connects"})
    custom_edges = [(str(dp2.id), str(dp1.id), "custom_edge", {})]

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

    result = await add_data_points([dp1, dp2], custom_edges=custom_edges)

    assert result == [dp1, dp2]
    graph_engine.add_nodes.assert_awaited_once()
    mock_index_nodes.assert_awaited_once()
    assert graph_engine.add_edges.await_count == 2
    expected_main_edges = ensure_default_edge_properties([edge1], nodes=[dp1, dp2])
    expected_custom_edges = ensure_default_edge_properties(custom_edges, nodes=[dp1, dp2])
    first_call_edges = graph_engine.add_edges.await_args_list[0].args[0]
    assert expected_main_edges[0] in first_call_edges
    assert expected_custom_edges[0] in first_call_edges
    assert graph_engine.add_edges.await_args_list[1].args[0] == expected_custom_edges
    assert mock_index_edges.await_count == 2


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_indexes_triplets_when_enabled(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    dp1 = SimplePoint(text="source")
    dp2 = SimplePoint(text="target")

    edge1 = (str(dp1.id), str(dp2.id), "relates", {"edge_text": "describes"})

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

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
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_empty_list(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

    result = await add_data_points([])

    assert result == []
    mock_get_graph.assert_not_called()
    graph_engine.add_nodes.assert_awaited_once_with([])


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_single_datapoint(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    dp = SimplePoint(text="single")
    mock_get_graph.side_effect = [([dp], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

    result = await add_data_points([dp])

    assert result == [dp]
    mock_get_graph.assert_called_once()
    mock_index_nodes.assert_awaited_once()


def test_entity_description_not_in_index_fields():
    from cognee.modules.engine.models import Entity, EntityType

    entity_type = EntityType(name="Person", description="A human being")
    entity = Entity(name="Alice", description="A software engineer", is_a=entity_type)
    text = _extract_embeddable_text_from_datapoint(entity)
    # description is stored but not indexed — only name is embedded
    assert "Alice" in text
    assert "A software engineer" not in text


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


def test_ensure_default_edge_properties_preserves_existing_defaults():
    # edge_text is supplied so the fallback path isn't invoked; this test
    # focuses on edge_object_id and feedback_weight surviving.
    edge = (
        "source",
        "target",
        "related_to",
        {
            "edge_object_id": "edge-id",
            "feedback_weight": 0.9,
            "edge_text": "source related to target",
        },
    )

    result = ensure_default_edge_properties([edge])

    properties = result[0][3]
    assert properties["edge_object_id"] == "edge-id"
    assert properties["feedback_weight"] == 0.9
    assert properties["edge_text"] == "source related to target"


@pytest.mark.parametrize(
    "properties",
    [{}, {"edge_text": None}, {"edge_text": ""}, {"edge_text": "   "}],
)
def test_ensure_default_edge_properties_falls_back_for_missing_or_blank_edge_text(properties):
    source = NamedPoint(name="Alice")
    target = NamedPoint(name="Acme")
    edge = (str(source.id), str(target.id), "works_at", properties)

    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["edge_text"] == "Alice works at Acme."


def test_ensure_default_edge_properties_preserves_nonblank_edge_text():
    edge = ("source", "target", "related_to", {"edge_text": "Alice works at Acme."})

    result = ensure_default_edge_properties([edge])

    assert result[0][3]["edge_text"] == "Alice works at Acme."


def test_ensure_default_edge_properties_uses_node_id_when_node_lookup_fails():
    # Nodes were not passed to ensure_default_edge_properties, so neither
    # endpoint can be resolved. Soft-fall back to the raw id so structural
    # edges still get a usable (if ugly) label.
    edge = ("source-node-id", "target-node-id", "related_to", {})

    result = ensure_default_edge_properties([edge])

    assert result[0][3]["edge_text"] == "source-node-id related to target-node-id."


def test_ensure_default_edge_properties_uses_type_name_when_label_cannot_be_derived():
    # Structural DataPoints (e.g. `Timestamp`) intentionally declare empty
    # `index_fields` and no `name`. The helper soft-falls back to the class
    # name so the graph still functions; a warning is emitted separately.
    source = UnlabelablePoint(payload="hello")
    target = NamedPoint(name="Acme")
    edge = (str(source.id), str(target.id), "related_to", {})

    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["edge_text"] == "UnlabelablePoint related to Acme."


def test_ensure_default_edge_properties_prefers_index_field_over_name():
    # When both `name` and a different `index_fields[0]` are declared, the
    # author's index_fields choice wins.
    source = IndexedNonNamePoint(name="Display Name", handle="@actual_handle")
    target = NamedPoint(name="Acme")
    edge = (str(source.id), str(target.id), "tagged", {})

    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["edge_text"] == "@actual_handle tagged Acme."


def test_ensure_default_edge_properties_falls_back_to_name_when_index_fields_missing():
    source = OnlyNamePoint(name="Alice")
    target = OnlyNamePoint(name="Bob")
    edge = (str(source.id), str(target.id), "knows", {})

    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["edge_text"] == "Alice knows Bob."


def test_ensure_default_edge_properties_prefers_title_when_name_is_unavailable():
    source = TitledPoint(title="Source Title")
    target = TitledPoint(title="Target Title")
    edge = (str(source.id), str(target.id), "references", {})

    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["edge_text"] == "Source Title references Target Title."


def test_ensure_default_edge_properties_uses_index_field_for_document_chunks():
    # DocumentChunk declares `metadata = {"index_fields": ["text"]}`, so the
    # fallback label uses the chunk's text (trimmed), per the index_fields
    # contract — no class-name special-casing in the helper.
    document = Document(
        name="Doc",
        raw_data_location="memory",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="Chunk text",
        chunk_size=10,
        chunk_index=3,
        cut_type="paragraph",
        is_part_of=document,
    )
    entity = NamedPoint(name="Alice")
    edge = (str(chunk.id), str(entity.id), "contains", {})

    result = ensure_default_edge_properties([edge], nodes=[chunk, entity])

    assert result[0][3]["edge_text"] == "Chunk text contains Alice."


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_with_empty_custom_edges(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    dp = SimplePoint(text="test")
    mock_get_graph.side_effect = [([dp], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

    result = await add_data_points([dp], custom_edges=[])

    assert result == [dp]
    assert graph_engine.add_edges.await_count == 1


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_hybrid_write_path(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    """When unified engine has HYBRID_WRITE, use add_nodes_with_vectors and add_edges_with_vectors."""
    from cognee.infrastructure.databases.unified.capabilities import EngineCapability

    dp1 = SimplePoint(text="first")
    dp2 = SimplePoint(text="second")

    edge1 = (str(dp1.id), str(dp2.id), "related_to", {"edge_text": "connects"})
    custom_edges = [(str(dp2.id), str(dp1.id), "custom_edge", {})]

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, vector_engine = _make_unified_mock()
    unified.has_capability = MagicMock(side_effect=lambda cap: cap == EngineCapability.HYBRID_WRITE)
    mock_get_unified.return_value = unified

    result = await add_data_points([dp1, dp2], custom_edges=custom_edges)

    assert result == [dp1, dp2]

    # Hybrid path: add_nodes_with_vectors called, not add_nodes
    graph_engine.add_nodes_with_vectors.assert_awaited_once()
    graph_engine.add_nodes.assert_not_awaited()

    # Hybrid path: add_edges_with_vectors called, not add_edges
    assert graph_engine.add_edges_with_vectors.await_count == 2
    graph_engine.add_edges.assert_not_awaited()

    # Standard index_data_points and index_graph_edges should NOT be called
    mock_index_nodes.assert_not_awaited()
    mock_index_edges.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "upsert_edges")
@patch.object(adp_module, "upsert_nodes")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_relational_upserts_happen_before_graph_and_vector_writes(
    mock_get_graph,
    mock_dedup,
    mock_get_unified,
    mock_upsert_nodes,
    mock_upsert_edges,
    mock_index_nodes,
    mock_index_edges,
):
    dp1 = SimplePoint(text="first")
    dp2 = SimplePoint(text="second")
    edge1 = (str(dp1.id), str(dp2.id), "related_to", {"edge_text": "connects"})
    custom_edges = [(str(dp2.id), str(dp1.id), "custom_edge", {})]

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, _vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified

    call_order = []

    async def _upsert_nodes(*_args, **_kwargs):
        call_order.append("upsert_nodes")

    async def _upsert_edges(*_args, **_kwargs):
        call_order.append("upsert_edges")

    async def _add_nodes(*_args, **_kwargs):
        call_order.append("graph_add_nodes")

    async def _add_edges(*_args, **_kwargs):
        call_order.append("graph_add_edges")

    async def _index_nodes(*_args, **_kwargs):
        call_order.append("index_nodes")

    async def _index_edges(*_args, **_kwargs):
        call_order.append("index_edges")

    mock_upsert_nodes.side_effect = _upsert_nodes
    mock_upsert_edges.side_effect = _upsert_edges
    graph_engine.add_nodes.side_effect = _add_nodes
    graph_engine.add_edges.side_effect = _add_edges
    mock_index_nodes.side_effect = _index_nodes
    mock_index_edges.side_effect = _index_edges

    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_name="cognify_pipeline",
        pipeline_run_id=uuid4(),
    )

    await add_data_points([dp1, dp2], custom_edges=custom_edges, ctx=ctx)

    first_graph_index = min(
        call_order.index("graph_add_nodes"),
        call_order.index("index_nodes"),
        call_order.index("graph_add_edges"),
        call_order.index("index_edges"),
    )
    assert call_order[:3] == ["upsert_nodes", "upsert_edges", "upsert_edges"]
    assert first_graph_index >= 3


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_cognify_config")
@patch.object(adp_module, "cluster_fuzzy_duplicate_entities")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_runs_fuzzy_dedup_before_exact_dedup(
    mock_get_graph,
    mock_dedup,
    mock_get_unified,
    mock_fuzzy_dedup,
    mock_get_config,
    mock_index_nodes,
    mock_index_edges,
):
    """Wiring check: fuzzy dedup runs (enabled by default), feeding its output
    into the existing exact-id dedup, with the unified engine's vector_engine."""
    dp1 = SimplePoint(text="first")
    dp2 = SimplePoint(text="second")
    edge1 = (str(dp1.id), str(dp2.id), "related_to", {"edge_text": "connects"})

    mock_get_graph.side_effect = [([dp1], [edge1]), ([dp2], [])]
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified
    mock_get_config.return_value = SimpleNamespace(
        fuzzy_entity_dedup=True, fuzzy_entity_dedup_threshold=0.95
    )

    fuzzy_nodes = [dp1]  # simulate dp2 being merged away
    fuzzy_edges = [edge1]
    mock_fuzzy_dedup.return_value = (fuzzy_nodes, fuzzy_edges, [{"duplicate_id": str(dp2.id)}])
    mock_dedup.side_effect = lambda n, e: (n, e)

    await add_data_points([dp1, dp2])

    mock_fuzzy_dedup.assert_awaited_once()
    call_args = mock_fuzzy_dedup.await_args
    assert call_args.args[0] == [dp1, dp2]  # nodes, pre-fuzzy-merge
    assert call_args.args[2] is vector_engine
    assert call_args.kwargs["similarity_threshold"] == 0.95

    # Exact dedup receives the fuzzy-merged output, not the raw pre-merge lists.
    mock_dedup.assert_called_once_with(fuzzy_nodes, fuzzy_edges)


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_cognify_config")
@patch.object(adp_module, "cluster_fuzzy_duplicate_entities")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_skips_fuzzy_dedup_when_disabled(
    mock_get_graph,
    mock_dedup,
    mock_get_unified,
    mock_fuzzy_dedup,
    mock_get_config,
    mock_index_nodes,
    mock_index_edges,
):
    dp = SimplePoint(text="only")
    mock_get_graph.side_effect = [([dp], [])]
    unified, graph_engine, vector_engine = _make_unified_mock()
    mock_get_unified.return_value = unified
    mock_get_config.return_value = SimpleNamespace(
        fuzzy_entity_dedup=False, fuzzy_entity_dedup_threshold=0.95
    )
    mock_dedup.side_effect = lambda n, e: (n, e)

    await add_data_points([dp])

    mock_fuzzy_dedup.assert_not_called()
    mock_dedup.assert_called_once_with([dp], [])
