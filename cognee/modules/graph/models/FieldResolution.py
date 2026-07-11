from cognee.infrastructure.engine import DataPoint

class FieldResolution(DataPoint):
    """Record of how a single field conflict was resolved."""
    field_name: str
    old_value: str
    new_value: str
    strategy: str
    # NO identity_fields -> UUID4 -> every resolution gets a unique id
    metadata: dict = {"index_fields": []}
