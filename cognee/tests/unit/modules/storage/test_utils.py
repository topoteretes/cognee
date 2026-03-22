# cognee/tests/unit/modules/storage/test_utils.py
import uuid
import pytest
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


# --- Test Models ---

class SimplePoint(DataPoint):
    name: str
    value: int


class NestedPoint(DataPoint):
    label: str
    child: SimplePoint


# --- Tests ---

def test_copy_model_basic():
    """copy_model returns a new instance with all fields preserved."""
    point = SimplePoint(id=uuid.uuid4(), name="test", value=42)
    result = copy_model(point)
    assert result.name == "test"
    assert result.value == 42


def test_copy_model_include_fields():
    """copy_model with include_fields only copies specified fields."""
    point = SimplePoint(id=uuid.uuid4(), name="test", value=42)
    result = copy_model(point, include_fields=["name"])
    assert result.name == "test"
    assert not hasattr(result, "value") or result.value is None


def test_copy_model_exclude_fields():
    """copy_model with exclude_fields omits the specified fields."""
    point = SimplePoint(id=uuid.uuid4(), name="test", value=42)
    result = copy_model(point, exclude_fields=["value"])
    assert result.name == "test"
    assert not hasattr(result, "value") or result.value is None


def test_copy_model_include_and_exclude():
    """copy_model with both include and exclude respects both constraints."""
    point = SimplePoint(id=uuid.uuid4(), name="test", value=42)
    result = copy_model(point, include_fields=[
                        "name", "value"], exclude_fields=["value"])
    assert result.name == "test"
    assert not hasattr(result, "value") or result.value is None


def test_copy_model_empty_fields():
    """copy_model handles a model with empty/default field values."""
    point = SimplePoint(id=uuid.uuid4(), name="", value=0)
    result = copy_model(point)
    assert result.name == ""
    assert result.value == 0


def test_copy_model_nested_object():
    """copy_model correctly copies a DataPoint containing a nested DataPoint."""
    child = SimplePoint(id=uuid.uuid4(), name="child", value=1)
    parent = NestedPoint(id=uuid.uuid4(), label="parent", child=child)
    result = copy_model(parent)
    assert result.label == "parent"
    assert result.child.name == "child"
    assert result.child.value == 1


def test_copy_model_returns_new_instance():
    """copy_model returns a different object, not the same reference."""
    point = SimplePoint(id=uuid.uuid4(), name="test", value=42)
    result = copy_model(point)
    assert result is not point
