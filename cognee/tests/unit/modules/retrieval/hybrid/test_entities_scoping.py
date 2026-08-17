"""Node-set scoping of the hybrid entity lane.

The entity vector search is already node-set filtered, but the one-hop
neighborhood used for edge bullets comes from an unfiltered graph round trip:
without scoping, edges to foreign-dataset neighbors (including edge_text
derived from foreign chunks) leak into the context. These tests pin the
fail-closed neighbor filter and the shared-entity description suppression.
"""

import pytest

from cognee.modules.retrieval.hybrid.entities import build_entities, _entity_from_result


class _GraphEngine:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self.calls = []

    async def get_neighborhood(self, entity_ids, depth=1):
        self.calls.append((tuple(entity_ids), depth))
        return self._nodes, self._edges


def _entity_hit(entity_id="entity-1", belongs_to_set=None, description="a vessel type"):
    payload = {"id": entity_id, "name": "psv", "type": "Entity", "description": description}
    if belongs_to_set is not None:
        payload["belongs_to_set"] = belongs_to_set
    return payload


def _mixed_neighborhood():
    """entity-1 has one same-set neighbor, one foreign neighbor, and one
    neighbor with no belongs_to_set at all (fail-closed case)."""
    nodes = [
        ("entity-1", {"name": "psv", "belongs_to_set": ["A"]}),
        ("same-set", {"name": "embarcacao", "belongs_to_set": ["A"]}),
        ("foreign", {"name": "op atlantico", "belongs_to_set": ["B"]}),
        ("untagged", {"name": "mystery"}),
    ]
    edges = [
        ("entity-1", "same-set", "related_to", {"edge_text": "psv related to embarcacao"}),
        ("foreign", "entity-1", "has_vessel_type", {"edge_text": "OP Atlantico has PSV"}),
        ("entity-1", "untagged", "mentions", {"edge_text": "psv mentions mystery"}),
    ]
    return nodes, edges


@pytest.mark.asyncio
async def test_scoped_search_drops_foreign_and_untagged_neighbors():
    nodes, edges = _mixed_neighborhood()
    engine = _GraphEngine(nodes, edges)

    entities = await build_entities(
        engine,
        [_entity_hit(belongs_to_set=["A"])],
        max_edges_per_entity=10,
        node_name=["A"],
    )

    bullet_texts = [edge["text"] for edge in entities[0]["edges"]]
    assert bullet_texts == ["psv related to embarcacao"]


@pytest.mark.asyncio
async def test_unscoped_search_keeps_all_neighbors():
    nodes, edges = _mixed_neighborhood()
    engine = _GraphEngine(nodes, edges)

    entities = await build_entities(
        engine,
        [_entity_hit(belongs_to_set=["A"])],
        max_edges_per_entity=10,
    )

    bullet_texts = {edge["text"] for edge in entities[0]["edges"]}
    assert bullet_texts == {
        "psv related to embarcacao",
        "OP Atlantico has PSV",
        "psv mentions mystery",
    }


@pytest.mark.asyncio
async def test_and_operator_requires_all_sets_on_neighbors():
    nodes = [
        ("entity-1", {"name": "psv", "belongs_to_set": ["A", "B"]}),
        ("both-sets", {"name": "shared", "belongs_to_set": ["A", "B"]}),
        ("only-a", {"name": "solo", "belongs_to_set": ["A"]}),
    ]
    edges = [
        ("entity-1", "both-sets", "related_to", {"edge_text": "shared bullet"}),
        ("entity-1", "only-a", "related_to", {"edge_text": "solo bullet"}),
    ]
    engine = _GraphEngine(nodes, edges)

    entities = await build_entities(
        engine,
        [_entity_hit(belongs_to_set=["A", "B"])],
        max_edges_per_entity=10,
        node_name=["A", "B"],
        node_name_filter_operator="AND",
    )

    bullet_texts = [edge["text"] for edge in entities[0]["edges"]]
    assert bullet_texts == ["shared bullet"]


@pytest.mark.asyncio
async def test_self_loop_edges_survive_scoping():
    nodes = [("entity-1", {"name": "psv", "belongs_to_set": ["A"]})]
    edges = [("entity-1", "entity-1", "self_ref", {"edge_text": "psv self reference"})]
    engine = _GraphEngine(nodes, edges)

    entities = await build_entities(
        engine,
        [_entity_hit(belongs_to_set=["A"])],
        max_edges_per_entity=10,
        node_name=["A"],
    )

    assert [edge["text"] for edge in entities[0]["edges"]] == ["psv self reference"]


def test_shared_entity_description_suppressed_when_scoped():
    entity = _entity_from_result(_entity_hit(belongs_to_set=["A", "B"]), node_name=["A"])
    assert entity["description"] is None


def test_single_set_entity_keeps_description_when_scoped():
    entity = _entity_from_result(_entity_hit(belongs_to_set=["A"]), node_name=["A"])
    assert entity["description"] == "a vessel type"


def test_untagged_entity_description_suppressed_when_scoped():
    entity = _entity_from_result(_entity_hit(belongs_to_set=None), node_name=["A"])
    assert entity["description"] is None


def test_descriptions_untouched_when_unscoped():
    shared = _entity_from_result(_entity_hit(belongs_to_set=["A", "B"]))
    untagged = _entity_from_result(_entity_hit(belongs_to_set=None))
    assert shared["description"] == "a vessel type"
    assert untagged["description"] == "a vessel type"
