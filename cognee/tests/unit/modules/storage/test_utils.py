"""Unit tests for storage utility functions (copy_model, get_own_properties)."""
import pytest
from uuid import UUID
from datetime import datetime, timezone

from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model, get_own_properties


class TestCopyModel:
    """Tests for the copy_model function."""

    def test_copy_model_basic(self):
        """Test basic DataPoint copying without modifications."""
        original = DataPoint(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            type="TestPoint",
        )

        copied = copy_model(original)

        assert copied.__name__ == "TestPoint"
        assert "id" in copied.model_fields
        assert "type" in copied.model_fields

    def test_copy_model_with_include_fields(self):
        """Test copying with additional included fields."""
        original = DataPoint(type="TestPoint")

        copied = copy_model(
            original,
            include_fields={"custom_field": (str, "default_value")}
        )

        assert "custom_field" in copied.model_fields
        assert copied.model_fields["custom_field"].default == "default_value"

    def test_copy_model_with_exclude_fields(self):
        """Test copying with excluded fields."""
        original = DataPoint(type="TestPoint")

        copied = copy_model(original, exclude_fields=["created_at", "updated_at"])

        assert "created_at" not in copied.model_fields
        assert "updated_at" not in copied.model_fields
        assert "type" in copied.model_fields

    def test_copy_model_with_include_and_exclude(self):
        """Test copying with both include and exclude fields."""
        original = DataPoint(type="TestPoint")

        copied = copy_model(
            original,
            include_fields={"new_field": (str, "value")},
            exclude_fields=["metadata"]
        )

        assert "new_field" in copied.model_fields
        assert "metadata" not in copied.model_fields
        assert "type" in copied.model_fields

    def test_copy_model_edge_case_empty_fields(self):
        """Test copying when excluding all fields."""
        # Note: This test documents expected behavior - excluding all fields
        # should still create a valid model (possibly empty)
        original = DataPoint(type="TestPoint")

        # Excluding 'type' which is required with a default
        copied = copy_model(original, exclude_fields=["type"])

        # The model should be created, even if minimal
        assert copied is not None
        assert copied.__name__ == "TestPoint"


class TestGetOwnProperties:
    """Tests for the get_own_properties function."""

    def test_get_own_properties_basic(self):
        """Test extracting own properties from a DataPoint."""
        data_point = DataPoint(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            type="TestPoint",
            version=2,
            source_pipeline="test_pipeline",
        )

        properties = get_own_properties(data_point)

        assert "id" in properties
        assert "type" in properties
        assert "version" in properties
        assert "source_pipeline" in properties

    def test_get_own_properties_excludes_nested(self):
        """Test that nested objects are excluded."""
        data_point = DataPoint(
            type="TestPoint",
            belongs_to_set=["item1", "item2"],
        )

        properties = get_own_properties(data_point)

        # belongs_to_set should be excluded as it's a list with specific types
        assert "belongs_to_set" not in properties

    def test_get_own_properties_preserves_primitives(self):
        """Test that primitive values are preserved."""
        data_point = DataPoint(
            type="TestPoint",
            version=1,
            topological_rank=5,
            source_user="test_user",
        )

        properties = get_own_properties(data_point)

        assert properties["version"] == 1
        assert properties["topological_rank"] == 5
        assert properties["source_user"] == "test_user"
