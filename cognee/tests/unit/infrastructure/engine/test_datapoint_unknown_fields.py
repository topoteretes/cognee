"""Stored nodes tolerate fields removed from the base model."""

from uuid import uuid4

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.Entity import Entity


@pytest.mark.parametrize("model", [DataPoint, Entity])
def test_stored_validity_field_is_ignored(model):
    node_id = uuid4()
    node = model.from_dict(
        {"id": node_id, "name": "Alice", "description": "A person.", "valid_to": 1}
    )
    assert node.id == node_id
    assert "valid_to" not in node.model_dump()
