from cognee.infrastructure.engine.models.DataPoint import DataPoint
from typing import List, Dict, Optional
from datetime import datetime


class DatabaseSchema(DataPoint):
    """Represents a complete database schema with sample data"""

    name: str
    database_type: str  # sqlite, postgres, etc.
    tables: str  # Reuse existing schema format from SqlAlchemyAdapter
    sample_data: str  # Limited examples per table
    description: str
    metadata: dict = {"index_fields": ["description", "name"]}


class SchemaTable(DataPoint):
    """Represents an individual table schema with relationships"""

    name: str
    columns: str  # Column definitions with types
    primary_key: Optional[str]
    foreign_keys: str  # Foreign key relationships
    sample_rows: str  # Max 3-5 example rows
    row_count_estimate: Optional[int]  # Actual table size
    description: str
    metadata: dict = {"index_fields": ["description", "name"]}


class SchemaRelationship(DataPoint):
    """Represents relationships between tables"""

    name: str
    source_table: str
    target_table: str
    relationship_type: str  # "foreign_key", "one_to_many", etc.
    source_column: str
    target_column: str
    description: str
    metadata: dict = {"index_fields": ["description", "name"]}
