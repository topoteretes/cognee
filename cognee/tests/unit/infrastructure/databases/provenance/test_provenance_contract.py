from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
    get_data_id_from_source_ref_key,
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.databases.unified import UnifiedStoreEngine
from cognee.tests.unit.infrastructure.databases.provenance.fakes import FakeGraphVectorStore


def test_source_ref_key_round_trip():
    dataset_id = uuid4()
    data_id = uuid4()

    source_ref_key = make_source_ref_key(dataset_id, data_id)

    assert source_ref_key == f"source_ref:v1:{dataset_id}:{data_id}"
    assert get_dataset_id_from_source_ref_key(source_ref_key) == dataset_id
    assert get_data_id_from_source_ref_key(source_ref_key) == data_id


def test_source_ref_key_rejects_unknown_prefix():
    source_ref_key = f"source_ref:v2:{uuid4()}:{uuid4()}"

    with pytest.raises(ValueError, match="Unsupported source ref key format"):
        get_dataset_id_from_source_ref_key(source_ref_key)


def test_source_run_ref_round_trip():
    pipeline_run_id = uuid4()
    source_ref_key = make_source_ref_key(uuid4(), uuid4())

    source_run_ref = make_source_run_ref(pipeline_run_id, source_ref_key)

    assert source_run_ref == f"source_run_ref:v1:{pipeline_run_id}:{source_ref_key}"
    assert get_pipeline_run_id_from_source_run_ref(source_run_ref) == pipeline_run_id
    assert get_source_ref_key_from_source_run_ref(source_run_ref) == source_ref_key


def test_source_run_ref_rejects_unknown_prefix():
    source_run_ref = f"source_run_ref:v2:{uuid4()}:{make_source_ref_key(uuid4(), uuid4())}"

    with pytest.raises(ValueError, match="Unsupported source run ref format"):
        get_pipeline_run_id_from_source_run_ref(source_run_ref)


def test_delete_data_dataclasses_are_stable_values():
    edge = EdgeIdentity("source", "target", "mentions")
    node_delete_data = NodeDeleteData(
        node_id="node",
        node_type="Entity",
        indexed_fields=["name"],
        node_properties={"name": "Germany"},
        source_ref_keys=["source_ref:v1:dataset:data"],
        source_dataset_ids=["dataset"],
        source_run_ids=["run"],
        source_run_refs=["source_run_ref:v1:run:source_ref:v1:dataset:data"],
    )
    edge_delete_data = EdgeDeleteData(
        edge=edge,
        edge_text="mentions",
        edge_properties={"weight": 1},
        source_ref_keys=node_delete_data.source_ref_keys,
        source_dataset_ids=node_delete_data.source_dataset_ids,
        source_run_ids=node_delete_data.source_run_ids,
        source_run_refs=node_delete_data.source_run_refs,
    )

    assert edge == EdgeIdentity("source", "target", "mentions")
    assert node_delete_data.node_properties["name"] == "Germany"
    assert edge_delete_data.edge == edge


def test_graph_provenance_marker_constants_import_cleanly():
    assert GRAPH_PROVENANCE_VERSION_KEY == "provenance_version"
    assert GRAPH_PROVENANCE_VERSION == "1"
    assert GRAPH_DELETE_MODE_KEY == "delete_mode"
    assert GRAPH_DELETE_MODE_GRAPH_PROVENANCE == "graph_native"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("attach_node_source_refs", (["node"], ["source_ref"])),
        (
            "attach_edge_source_refs",
            ([EdgeIdentity("source", "target", "mentions")], ["source_ref"]),
        ),
        ("remove_node_source_refs", (["node"], ["source_ref"])),
        (
            "remove_edge_source_refs",
            ([EdgeIdentity("source", "target", "mentions")], ["source_ref"]),
        ),
        ("delete_edge_triples", ([EdgeIdentity("source", "target", "mentions")],)),
        ("get_node_delete_data", (["node"],)),
        ("get_edge_delete_data", ([EdgeIdentity("source", "target", "mentions")],)),
        ("find_nodes_by_source_ref", ("source_ref",)),
        ("find_edges_by_source_ref", ("source_ref",)),
        ("find_node_source_refs_by_dataset", ("dataset",)),
        ("find_edge_source_refs_by_dataset", ("dataset",)),
        ("find_node_source_refs_by_pipeline_run", ("run",)),
        ("find_edge_source_refs_by_pipeline_run", ("run",)),
        ("set_graph_metadata", ({"provenance_version": "1"},)),
        ("get_graph_metadata", ()),
    ],
)
async def test_graph_provenance_defaults_raise_typed_error(method_name, args):
    with pytest.raises(UnsupportedProvenanceCapability):
        await getattr(GraphDBInterface, method_name)(object(), *args)


@pytest.mark.asyncio
async def test_unified_graph_vector_defaults_raise_typed_error():
    engine = UnifiedStoreEngine()

    with pytest.raises(UnsupportedProvenanceCapability):
        await engine.delete_by_source_ref("source_ref")

    with pytest.raises(UnsupportedProvenanceCapability):
        await engine.delete_by_dataset_id(str(uuid4()))

    with pytest.raises(UnsupportedProvenanceCapability):
        await engine.rollback_by_pipeline_run_id(str(uuid4()))


@pytest.mark.asyncio
async def test_part2_fake_store_implements_unified_contract():
    store = FakeGraphVectorStore()
    source_ref_key = make_source_ref_key(uuid4(), uuid4())
    dataset_id = str(get_dataset_id_from_source_ref_key(source_ref_key))
    pipeline_run_id = str(uuid4())

    await store.delete_by_source_ref(source_ref_key)
    await store.delete_by_dataset_id(dataset_id)
    await store.rollback_by_pipeline_run_id(pipeline_run_id)

    assert store.deleted_source_refs == [source_ref_key]
    assert store.deleted_dataset_ids == [dataset_id]
    assert store.rolled_back_pipeline_run_ids == [pipeline_run_id]


def test_unsupported_provenance_capability_does_not_log_by_default(monkeypatch):
    calls = []

    def fake_error(message):
        calls.append(message)

    monkeypatch.setattr("cognee.exceptions.exceptions.logger.error", fake_error)

    error = UnsupportedProvenanceCapability()

    assert error.status_code == 501
    assert calls == []
