"""Unit tests for the Pydantic-native export (GraphSnapshot). Pure: no DB, no LLM."""

from uuid import uuid4

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.migration.snapshot import (
    GraphSnapshot,
    build_snapshot,
    datapoint_registry,
    rehydrate_node,
)


class PetDataPoint(DataPoint):
    """Custom user model: must be picked up by the registry automatically."""

    name: str
    species: str = "dog"
    toys: list = []
    metadata: dict = {"index_fields": ["name"]}


def _entity_props(name="Alice", **extra):
    return {
        "id": str(uuid4()),
        "type": "Entity",
        "name": name,
        "description": f"{name} is a person",
        **extra,
    }


class TestRegistry:
    def test_core_models_registered(self):
        registry = datapoint_registry()
        assert registry["Entity"] is Entity
        assert registry["EntityType"] is EntityType

    def test_custom_models_registered_on_import(self):
        registry = datapoint_registry()
        assert registry["PetDataPoint"] is PetDataPoint
        # Module-qualified key disambiguates name collisions.
        assert registry[f"{PetDataPoint.__module__}.PetDataPoint"] is PetDataPoint


class TestRehydrateNode:
    def test_known_type_returns_real_instance(self):
        node = rehydrate_node(_entity_props())
        assert isinstance(node, Entity)
        assert node.name == "Alice"
        assert node.type == "Entity"

    def test_extra_bookkeeping_keys_ignored(self):
        node = rehydrate_node(_entity_props(some_internal_marker="x"))
        assert isinstance(node, Entity)

    def test_unknown_type_degrades_to_dynamic_model(self):
        node = rehydrate_node(
            {"id": str(uuid4()), "type": "AlienModel", "name": "Zorg", "tentacles": 7}
        )
        assert isinstance(node, DataPoint)
        assert node.type == "AlienModel"
        assert node.tentacles == 7  # extra="allow" keeps every property

    def test_json_string_metadata_parsed(self):
        node = rehydrate_node(_entity_props(metadata='{"index_fields": ["name"]}'))
        assert isinstance(node, Entity)
        assert node.metadata["index_fields"] == ["name"]


class TestGraphSnapshot:
    def _snapshot(self, link_relations=False):
        alice = _entity_props("Alice")
        person = {
            "id": str(uuid4()),
            "type": "EntityType",
            "name": "Person",
            "description": "A person type",
        }
        pet = {
            "id": str(uuid4()),
            "type": "PetDataPoint",
            "name": "Rex",
            "species": "dog",
        }
        nodes = [(p["id"], p) for p in (alice, person, pet)]
        edges = [
            (alice["id"], person["id"], "is_a", {"edge_text": "Alice is a person"}),
            (alice["id"], pet["id"], "unknown_relation", {}),
        ]
        return build_snapshot(nodes, edges, "main_dataset", "ds-1", link_relations=link_relations)

    def test_nodes_are_typed_instances(self):
        snapshot = self._snapshot()
        assert isinstance(snapshot.find(name="Alice")[0], Entity)
        assert isinstance(snapshot.find(name="Rex")[0], PetDataPoint)
        assert len(snapshot.nodes_of_type(Entity)) == 1
        assert len(snapshot.nodes_of_type("EntityType")) == 1

    def test_json_roundtrip_preserves_subclasses_and_fields(self):
        snapshot = self._snapshot()
        blob = snapshot.model_dump_json()
        restored = GraphSnapshot.model_validate_json(blob)

        assert restored.dataset_name == "main_dataset"
        assert len(restored.nodes) == 3
        rex = restored.find(name="Rex")[0]
        assert isinstance(rex, PetDataPoint)  # subclass survives the round trip
        assert rex.species == "dog"  # subclass-only field survives too
        alice = restored.find(name="Alice")[0]
        assert isinstance(alice, Entity)
        assert alice.description == "Alice is a person"
        assert [e.relationship for e in restored.edges] == ["is_a", "unknown_relation"]

    def test_link_relations_attaches_declared_fields(self):
        snapshot = self._snapshot(link_relations=True)
        alice = snapshot.find(name="Alice")[0]
        # is_a is a declared Entity field -> re-attached as an object reference.
        assert isinstance(alice.is_a, EntityType)
        assert alice.is_a.name == "Person"

    def test_link_relations_skips_undeclared_fields(self):
        snapshot = self._snapshot(link_relations=True)
        alice = snapshot.find(name="Alice")[0]
        assert not hasattr(alice, "unknown_relation")

    def test_save_and_load(self, tmp_path):
        snapshot = self._snapshot()
        path = tmp_path / "memory.json"
        snapshot.save(path)
        restored = GraphSnapshot.load(path)
        assert isinstance(restored.find(name="Alice")[0], Entity)
