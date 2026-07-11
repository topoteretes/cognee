import pytest
from uuid import UUID

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import deduplicate_nodes_and_edges

def test_cross_type_different_ids_at_construction():
    """Person Jordan and Country Jordan get different UUIDs at construction."""
    person_id = Entity.id_for("jordan", "person")
    country_id = Entity.id_for("jordan", "country")

    assert person_id != country_id

def test_same_type_same_id_at_construction():
    """Two 'Apple Inc.' companies share an id -> merge correctly."""
    id1 = Entity.id_for("apple_inc", "company")
    id2 = Entity.id_for("apple_inc", "company")
    
    assert id1 == id2

def test_no_type_falls_back_to_name_only():
    """Entity with no type uses the old name-only hash -> no regression."""
    id_with_type = Entity.id_for("jordan", "unknown")
    id_no_type = Entity.id_for("jordan")
    
    assert id_with_type != id_no_type

def test_deduplicate_nodes_and_edges_merges_same_id():
    # Setup
    e1 = Entity(id=Entity.id_for("apple_inc", "company"), name="Apple Inc.", description="Tech company")
    e2 = Entity(id=Entity.id_for("apple_inc", "company"), name="Apple", description="A tech company")
    
    nodes = [e1, e2]
    
    final_nodes, final_edges, merge_records = deduplicate_nodes_and_edges(nodes, [])
    
    assert len(final_nodes) == 1
    assert len(merge_records) == 1
    assert final_nodes[0].name == "apple_inc" # Canonicalized name
