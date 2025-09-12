from cognee.infrastructure.engine.models.DataPoint import DataPoint
from typing import List, Dict, Optional
from datetime import datetime

class DatabaseSchema(DataPoint):
    """Represents a complete database schema with sample data"""
    schema_name: str
    database_type: str  # sqlite, postgres, etc.
    tables: Dict[str, Dict]  # Reuse existing schema format from SqlAlchemyAdapter
    sample_data: Dict[str, List[Dict]]  # Limited examples per table
    extraction_timestamp: datetime
    metadata: dict = {"index_fields": ["schema_name", "database_type"]}

class SchemaTable(DataPoint):
    """Represents an individual table schema with relationships"""
    table_name: str
    schema_name: str
    columns: List[Dict]  # Column definitions with types
    primary_key: Optional[str]
    foreign_keys: List[Dict]  # Foreign key relationships
    sample_rows: List[Dict]  # Max 3-5 example rows
    row_count_estimate: Optional[int]  # Actual table size
    metadata: dict = {"index_fields": ["table_name", "schema_name"]}

class SchemaRelationship(DataPoint):
    """Represents relationships between tables"""
    source_table: str
    target_table: str
    relationship_type: str  # "foreign_key", "one_to_many", etc.
    source_column: str
    target_column: str
    metadata: dict = {"index_fields": ["source_table", "target_table"]}