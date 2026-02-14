
import pytest
from uuid import UUID
from cognee.infrastructure.engine.models.DataPoint import DataPoint

class TestDataPoint:
    def test_datapoint_initialization(self):
        """Test that a DataPoint can be initialized with default values."""
        dp = DataPoint()
        assert isinstance(dp.id, UUID)
        assert dp.version == 1
        assert dp.metadata == {"index_fields": []}
        assert dp.type == "DataPoint"

    def test_to_json_returns_string(self):
        """Test that to_json() returns a string."""
        dp = DataPoint()
        json_output = dp.to_json()
        assert isinstance(json_output, str)
        assert len(json_output) > 0

    def test_from_json_roundtrip(self):
        """Test that an object serialized to JSON can be deserialized back to an equivalent object."""
        original_dp = DataPoint(ontology_valid=True)
        json_output = original_dp.to_json()
        
        restored_dp = DataPoint.from_json(json_output)
        
        assert isinstance(restored_dp, DataPoint)
        assert restored_dp.id == original_dp.id
        assert restored_dp.ontology_valid == original_dp.ontology_valid
        assert restored_dp.created_at == original_dp.created_at
        
        # Verify full model dump equality
        assert restored_dp.model_dump() == original_dp.model_dump()

    def test_to_dict_returns_dict(self):
        """Test that to_dict() returns a dictionary."""
        dp = DataPoint()
        dict_output = dp.to_dict()
        assert isinstance(dict_output, dict)
        assert dict_output["id"] == dp.id
        assert dict_output["version"] == dp.version

    def test_from_dict_roundtrip(self):
        """Test that an object converted to dict can be restored back to an equivalent object."""
        original_dp = DataPoint(topological_rank=5)
        dict_output = original_dp.to_dict()
        
        restored_dp = DataPoint.from_dict(dict_output)
        
        assert isinstance(restored_dp, DataPoint)
        assert restored_dp.id == original_dp.id
        assert restored_dp.topological_rank == original_dp.topological_rank
        
        # Verify full model dump equality
        assert restored_dp.model_dump() == original_dp.model_dump()
