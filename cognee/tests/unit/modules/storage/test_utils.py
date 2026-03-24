"""Unit tests for cognee/modules/storage/utils (copy_model, get_own_properties, JSONEncoder)."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder, copy_model, get_own_properties


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SamplePoint(DataPoint):
    """Minimal DataPoint subclass for testing."""
    name: str = "default"
    score: float = 0.0


class NestedPoint(DataPoint):
    """DataPoint with a nested DataPoint field."""
    label: str = "nested"
    children: Optional[List[DataPoint]] = None


# ---------------------------------------------------------------------------
# copy_model – basic
# ---------------------------------------------------------------------------

class TestCopyModelBasic:
    def test_copy_preserves_fields(self):
        """copy_model with no include/exclude should keep all fields."""
        copied = copy_model(SamplePoint)
        field_names = set(copied.model_fields.keys())
        assert "name" in field_names
        assert "score" in field_names
        # inherited DataPoint fields should also be present
        assert "id" in field_names

    def test_copy_returns_new_class(self):
        """The returned model should be a distinct class, not the original."""
        copied = copy_model(SamplePoint)
        assert copied is not SamplePoint


# ---------------------------------------------------------------------------
# copy_model – include_fields
# ---------------------------------------------------------------------------

class TestCopyModelInclude:
    def test_include_adds_extra_field(self):
        """include_fields should add a new field to the copied model."""
        copied = copy_model(SamplePoint, include_fields={"extra": (str, "hello")})
        field_names = set(copied.model_fields.keys())
        assert "extra" in field_names
        assert "name" in field_names

    def test_include_override_existing_field(self):
        """include_fields can override an existing field's default."""
        copied = copy_model(SamplePoint, include_fields={"name": (int, 42)})
        # The overridden field should appear with the new type
        assert copied.model_fields["name"].annotation is int


# ---------------------------------------------------------------------------
# copy_model – exclude_fields
# ---------------------------------------------------------------------------

class TestCopyModelExclude:
    def test_exclude_removes_field(self):
        """exclude_fields should remove the specified field."""
        copied = copy_model(SamplePoint, exclude_fields=["score"])
        field_names = set(copied.model_fields.keys())
        assert "score" not in field_names
        assert "name" in field_names

    def test_exclude_multiple_fields(self):
        """Multiple fields can be excluded at once."""
        copied = copy_model(SamplePoint, exclude_fields=["name", "score"])
        field_names = set(copied.model_fields.keys())
        assert "name" not in field_names
        assert "score" not in field_names


# ---------------------------------------------------------------------------
# copy_model – include + exclude combined
# ---------------------------------------------------------------------------

class TestCopyModelCombined:
    def test_include_and_exclude_together(self):
        """Exclude a field and include a new one simultaneously."""
        copied = copy_model(
            SamplePoint,
            include_fields={"tag": (str, "v1")},
            exclude_fields=["score"],
        )
        field_names = set(copied.model_fields.keys())
        assert "tag" in field_names
        assert "score" not in field_names
        assert "name" in field_names


# ---------------------------------------------------------------------------
# get_own_properties
# ---------------------------------------------------------------------------

class TestGetOwnProperties:
    def test_simple_properties(self):
        """Should return scalar fields, excluding metadata."""
        point = SamplePoint(name="test", score=3.14)
        props = get_own_properties(point)
        assert props["name"] == "test"
        assert props["score"] == 3.14
        assert "metadata" not in props

    def test_excludes_nested_datapoints(self):
        """DataPoint-typed fields should be excluded."""
        child = SamplePoint(name="child")
        parent = NestedPoint(label="parent", children=[child])
        props = get_own_properties(parent)
        assert "label" in props
        assert "children" not in props


# ---------------------------------------------------------------------------
# JSONEncoder
# ---------------------------------------------------------------------------

class TestJSONEncoder:
    def test_datetime_encoding(self):
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = json.dumps({"ts": dt}, cls=JSONEncoder)
        assert "2026-01-01" in result

    def test_uuid_encoding(self):
        uid = uuid4()
        result = json.dumps({"id": uid}, cls=JSONEncoder)
        assert str(uid) in result

    def test_decimal_encoding(self):
        d = Decimal("3.14")
        result = json.dumps({"val": d}, cls=JSONEncoder)
        assert "3.14" in result
