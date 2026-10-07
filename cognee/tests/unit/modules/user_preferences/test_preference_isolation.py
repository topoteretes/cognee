"""Unit tests for Phase 1 of user preferences: marker, projection filter, node id.

Covers the deterministic pieces only — no graph database, no LLM:
- ``is_internal_node`` truth table, including a ``UserPreference`` model dump.
- ``CogneeGraph._process_nodes_and_edges`` drops an internal node AND every
  edge out of it, even though the code has no edge-level check.
- ``preference_node_id`` determinism (rule 7: one helper, both sides identical).
"""

from uuid import UUID

from cognee.infrastructure.engine import INTERNAL_PROPERTY, is_internal_node
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.user_preferences import UserPreference, preference_node_id
from cognee.tasks.graph.detect_contradictions import _node_names
from cognee.tasks.memify.get_triplet_datapoints import _process_single_triplet


class TestIsInternalNode:
    def test_none_is_not_internal(self):
        assert is_internal_node(None) is False

    def test_empty_dict_is_not_internal(self):
        assert is_internal_node({}) is False

    def test_normal_node_properties_are_not_internal(self):
        assert is_internal_node({"name": "Postgres", "type": "Entity"}) is False

    def test_explicit_false_is_not_internal(self):
        assert is_internal_node({INTERNAL_PROPERTY: False}) is False

    def test_user_preference_dump_is_internal(self):
        preference = UserPreference(user_id="user-1", dataset_id="ds-1")
        assert is_internal_node(preference.model_dump()) is True

    def test_marker_is_a_class_default(self):
        # Constructing the node is what sets the marker — no writer passes it.
        assert UserPreference(user_id="u", dataset_id="d").is_internal is True


class TestProjectionFilter:
    def _build_graph(self, nodes_data, edges_data):
        graph = CogneeGraph()
        graph._process_nodes_and_edges(
            nodes_data,
            edges_data,
            node_properties_to_project=["name", "type"],
            edge_properties_to_project=["relationship_name"],
            directed=True,
            node_dimension=1,
            edge_dimension=1,
            triplet_distance_penalty=6.5,
        )
        return graph

    def test_internal_node_and_its_edges_are_absent(self):
        nodes_data = [
            ("node-a", {"name": "A", "type": "Entity"}),
            ("node-b", {"name": "B", "type": "Entity"}),
            ("pref-1", {"type": "UserPreference", INTERNAL_PROPERTY: True}),
        ]
        edges_data = [
            ("node-a", "node-b", "related_to", {"relationship_name": "related_to"}),
            # Edge out of the internal node: dropped with no edge-level check,
            # because projection skips edges whose endpoints were not projected.
            ("pref-1", "node-a", "prefers", {"relationship_name": "prefers"}),
        ]

        graph = self._build_graph(nodes_data, edges_data)

        assert graph.get_node("pref-1") is None
        assert set(graph.nodes.keys()) == {"node-a", "node-b"}
        assert len(graph.edges) == 1
        relationship_types = {edge.attributes["relationship_type"] for edge in graph.edges}
        assert relationship_types == {"related_to"}

    def test_graph_without_internal_nodes_is_untouched(self):
        nodes_data = [
            ("node-a", {"name": "A", "type": "Entity"}),
            ("node-b", {"name": "B", "type": "Entity"}),
        ]
        edges_data = [
            ("node-a", "node-b", "related_to", {"relationship_name": "related_to"}),
        ]

        graph = self._build_graph(nodes_data, edges_data)

        assert set(graph.nodes.keys()) == {"node-a", "node-b"}
        assert len(graph.edges) == 1


class TestPreferenceNodeId:
    def test_deterministic(self):
        first = preference_node_id("user-1", "dataset-1")
        second = preference_node_id("user-1", "dataset-1")
        assert isinstance(first, UUID)
        assert first == second

    def test_distinct_per_user_and_dataset(self):
        base = preference_node_id("user-1", "dataset-1")
        assert preference_node_id("user-2", "dataset-1") != base
        assert preference_node_id("user-1", "dataset-2") != base


class TestTripletEmbeddingFilter:
    def test_internal_endpoint_skips_triplet(self):
        triplet = {
            "start_node": {"id": "pref-1", "type": "UserPreference", INTERNAL_PROPERTY: True},
            "end_node": {"id": "chunk-1", "type": "DocumentChunk", "text": "some text"},
            "relationship_properties": {"relationship_name": "prefers"},
        }

        triplet_obj, error_msg = _process_single_triplet(triplet, {"DocumentChunk": ["text"]}, 0, 0)

        assert triplet_obj is None
        assert "internal node endpoint" in error_msg

    def test_normal_triplet_still_processed(self):
        triplet = {
            "start_node": {"id": "chunk-1", "type": "DocumentChunk", "text": "some text"},
            "end_node": {"id": "chunk-2", "type": "DocumentChunk", "text": "other text"},
            "relationship_properties": {"relationship_name": "related_to"},
        }

        triplet_obj, error_msg = _process_single_triplet(triplet, {"DocumentChunk": ["text"]}, 0, 0)

        assert error_msg is None
        assert triplet_obj is not None


class TestContradictionNodeNames:
    def test_internal_node_omitted_from_name_map(self):
        nodes = [
            ("node-a", {"name": "Postgres"}),
            ("pref-1", {"name": "named internal node", INTERNAL_PROPERTY: True}),
            ("node-b", {}),
        ]

        assert _node_names(nodes) == {"node-a": "Postgres"}
