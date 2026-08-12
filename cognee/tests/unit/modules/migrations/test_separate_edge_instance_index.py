import asyncio
from copy import deepcopy

import pytest

from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions.separate_edge_instance_index import downgrade, migrate


class FakeGraphEngine:
    def __init__(self):
        self.edges = [
            ["a", "b", "depends_on", {"edge_text": "A depends on B"}],
            [
                "c",
                "d",
                "depends_on",
                {
                    "edge_text": "C depends on D",
                    "edge_object_id": generate_edge_object_id("c", "d", "depends_on"),
                },
            ],
            ["e", "f", "contains", {"edge_text": "E contains F"}],
            ["placeholder", "placeholder", "SELF", {}],
        ]
        self.batch_requests = []
        self.repaired_batches = []

    async def get_triplets_batch(self, offset, limit):
        self.batch_requests.append((offset, limit))
        return [
            {
                "start_node": {"id": source},
                "relationship_properties": {
                    "relationship_name": relationship_name,
                    "properties": deepcopy(properties),
                },
                "end_node": {"id": target},
            }
            for source, target, relationship_name, properties in self.edges[offset : offset + limit]
        ]

    async def add_edges(self, edges):
        self.repaired_batches.append(deepcopy(edges))
        by_key = {(edge[0], edge[1], edge[2]): edge for edge in self.edges}
        for edge in edges:
            by_key[(edge[0], edge[1], edge[2])][3] = deepcopy(edge[3])


class FakeVectorEngine:
    def __init__(self):
        self.collections = {
            "EdgeType_relationship_name": {
                "corrupt-type": {"relationship_name": "obsolete", "number_of_edges": 99},
            },
            "EdgeInstance_text": {
                "corrupt-instance": {"text": "wrong"},
            },
            "Triplet_text": {"triplet-stays": {"text": "untouched"}},
        }
        self.fail_replacement_number = None
        self.replacement_calls = 0

    async def replace_index_data_points(self, index_name, index_property_name, batches):
        self.replacement_calls += 1
        if self.replacement_calls == self.fail_replacement_number:
            raise RuntimeError("injected replacement failure")
        replacement = {}
        async for batch in batches:
            assert len(batch) <= 1000
            for point in batch:
                payload = point.model_dump()
                replacement[str(point.id)] = {
                    key: payload[key]
                    for key in (
                        "text",
                        "relationship_name",
                        "number_of_edges",
                        "source_node_id",
                        "target_node_id",
                    )
                    if key in payload
                }
        self.collections[f"{index_name}_{index_property_name}"] = replacement


def _expected_instance_ids():
    return {
        generate_edge_object_id("a", "b", "depends_on"),
        generate_edge_object_id("c", "d", "depends_on"),
        generate_edge_object_id("e", "f", "contains"),
    }


def test_edge_object_id_derivation_is_pinned_for_this_migration():
    assert generate_edge_object_id("a", "b", "depends_on") == (
        "bd2e7f06-6cef-5912-944a-114b4796bcde"
    )


def test_migration_rebuilds_exact_graph_derived_indexes_and_is_idempotent():
    graph = FakeGraphEngine()
    vector = FakeVectorEngine()
    context = MigrationContext(graph_engine=graph, vector_engine=vector)

    asyncio.run(migrate(context))

    assert set(vector.collections["EdgeType_relationship_name"]) == {
        str(EdgeType.id_for("depends_on")),
        str(EdgeType.id_for("contains")),
    }
    depends_on = vector.collections["EdgeType_relationship_name"][
        str(EdgeType.id_for("depends_on"))
    ]
    assert depends_on["number_of_edges"] == 2
    assert set(vector.collections["EdgeInstance_text"]) == _expected_instance_ids()
    assert vector.collections["Triplet_text"] == {"triplet-stays": {"text": "untouched"}}
    assert graph.edges[0][3]["edge_object_id"] == generate_edge_object_id("a", "b", "depends_on")
    first_state = deepcopy(vector.collections)

    asyncio.run(migrate(context))

    assert vector.collections == first_state


def test_retry_after_second_replacement_failure_converges_without_duplicates():
    graph = FakeGraphEngine()
    vector = FakeVectorEngine()
    vector.fail_replacement_number = 2
    context = MigrationContext(graph_engine=graph, vector_engine=vector)

    with pytest.raises(RuntimeError, match="injected replacement failure"):
        asyncio.run(migrate(context))

    assert set(vector.collections["EdgeType_relationship_name"]) == {
        str(EdgeType.id_for("depends_on")),
        str(EdgeType.id_for("contains")),
    }
    assert set(vector.collections["EdgeInstance_text"]) == {"corrupt-instance"}

    vector.fail_replacement_number = None
    asyncio.run(migrate(context))

    assert set(vector.collections["EdgeInstance_text"]) == _expected_instance_ids()
    assert len(vector.collections["EdgeInstance_text"]) == 3


def test_downgrade_removes_instances_and_rebuilds_legacy_edge_text_rows():
    graph = FakeGraphEngine()
    vector = FakeVectorEngine()

    asyncio.run(downgrade(MigrationContext(graph_engine=graph, vector_engine=vector)))

    assert vector.collections["EdgeInstance_text"] == {}
    assert set(vector.collections["EdgeType_relationship_name"]) == {
        str(EdgeType.id_for("A depends on B")),
        str(EdgeType.id_for("C depends on D")),
        str(EdgeType.id_for("E contains F")),
    }
    assert vector.collections["Triplet_text"] == {"triplet-stays": {"text": "untouched"}}


def test_unpaginated_adapter_loads_full_graph_once_and_still_streams_replacements():
    class FallbackGraph(FakeGraphEngine):
        def __init__(self):
            super().__init__()
            self.get_graph_data_calls = 0

        async def get_triplets_batch(self, offset, limit):
            raise NotImplementedError

        async def get_graph_data(self):
            self.get_graph_data_calls += 1
            return [], deepcopy(self.edges)

    graph = FallbackGraph()
    vector = FakeVectorEngine()

    asyncio.run(migrate(MigrationContext(graph_engine=graph, vector_engine=vector)))

    assert graph.get_graph_data_calls == 1
    assert set(vector.collections["EdgeInstance_text"]) == _expected_instance_ids()
