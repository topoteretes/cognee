import pytest

from cognee.modules.engine.models import Entity
from cognee.modules.graph.utils.merge_policy import MergePolicy, MergeStrategy

def test_merge_nodes_longest_description():
    policy = MergePolicy({"description": MergeStrategy.LONGEST})
    
    survivor = Entity(id=Entity.id_for("test"), name="test", description="Short")
    absorbed = Entity(id=Entity.id_for("test"), name="test", description="A much longer description")
    
    result, resolutions = policy.merge_nodes(survivor, absorbed)
    
    assert result.description == "A much longer description"
    assert len(resolutions) == 1
    assert resolutions[0].field_name == "description"
    assert resolutions[0].new_value == "A much longer description"
    assert resolutions[0].old_value == "Short"

def test_merge_with_existing_dict():
    policy = MergePolicy({"description": MergeStrategy.LONGEST})
    
    incoming = Entity(id=Entity.id_for("test"), name="test", description="New description that is very long")
    existing_props = {
        "id": str(incoming.id),
        "name": "test",
        "description": "Old short",
    }
    
    resolutions = policy.merge_with_existing(incoming, existing_props)
    
    assert incoming.description == "New description that is very long"
    assert len(resolutions) == 1
    assert resolutions[0].new_value == "New description that is very long"
