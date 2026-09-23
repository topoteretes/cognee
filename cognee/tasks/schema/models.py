from datetime import datetime

from cognee.infrastructure.engine.models.DataPoint import DataPoint


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
    primary_key: str | None
    foreign_keys: str  # Foreign key relationships
    sample_rows: str  # Max 3-5 example rows
    row_count_estimate: int | None  # Actual table size
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


class SchemaColumn(DataPoint):
    """One column of a source table, written only when it realizes an ontology property.

    Columns are not materialised wholesale (a wide schema would drown the graph in
    them); schema alignment creates a node for a column exactly when it can say what
    business property the column carries (``cust_id`` realizes ``hasCustomerId``).
    """

    name: str
    table_name: str
    data_type: str = ""
    description: str
    metadata: dict = {"index_fields": ["description", "name"]}
