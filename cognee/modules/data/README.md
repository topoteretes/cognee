# Data Management Module

This module provides utilities for managing data lifecycle in Cognee, including automatic cleanup of unused data and access tracking across all graph databases.

## Overview

The data module consists of two main components:

1. **Cleanup Unused Data** (`cleanup_unused_data.py`) - Schema-agnostic data cleanup based on access patterns
2. **Access Tracking** (`access_tracking.py`) - Track when data entities are accessed

## Key Features

✅ **Schema-Agnostic**: Works with both default and custom graph schemas
✅ **Dynamic Table Discovery**: Automatically discovers tables with `last_accessed` column
✅ **Safe Operations**: Supports dry-run mode to preview deletions
✅ **Comprehensive Statistics**: Get detailed usage statistics for all tracked tables
✅ **Bulk Operations**: Efficiently update access timestamps for multiple entities

## Architecture

### Schema-Agnostic Design

The module uses SQLAlchemy metadata reflection to dynamically discover tables at runtime, eliminating the need for hardcoded table names. This ensures compatibility with:

- Default Cognee knowledge graph (tables: `document_chunks`, `entities`, `summaries`, `associations`, `metadata`)
- Custom user-defined graphs with any schema name
- Future table additions without code changes

### How It Works

1. **Table Discovery**: Uses `MetaData.reflect()` to discover all tables in a schema
2. **Column Detection**: Filters tables that have a `last_accessed` column
3. **Dynamic Queries**: Builds SQLAlchemy queries using discovered table objects
4. **Safe Execution**: All operations are transactional with rollback support

## Usage

### Cleanup Unused Data

```python
from cognee.tasks.cleanup_unused_data import cleanup_unused_data

# Preview what would be deleted (dry run) - default graph
result = await cleanup_unused_data(
    days_threshold=30,
    dry_run=True
)

# Actually delete old data - default graph
result = await cleanup_unused_data(
    days_threshold=60,
    dry_run=False
)

# Cleanup custom graph
result = await cleanup_unused_data(
    days_threshold=30,
    dry_run=False,
    schema='my_custom_graph'
)

print(f"Status: {result.status}")
print(f"Deleted counts: {result.deleted_counts}")
print(f"Errors: {result.errors}")
```

### Access Tracking

```python
from cognee.modules.data.access_tracking import (
    mark_entity_accessed,
    bulk_update_last_accessed
)
from uuid import UUID

# Mark a single entity as accessed
await mark_entity_accessed(
    session=session,
    entity_id=UUID('...'),
    table_name='document_chunks'
)

# Bulk update for multiple tables
await bulk_update_last_accessed(
    session=session,
    updates=[
        {'table_name': 'document_chunks', 'entity_ids': [id1, id2]},
        {'table_name': 'entities', 'entity_ids': [id3, id4]}
    ]
)

# Custom graph access tracking
await mark_entity_accessed(
    session=session,
    entity_id=UUID('...'),
    table_name='custom_nodes',
    schema='my_custom_graph'
)
```

### Get Statistics

```python
from cognee.tasks.cleanup_unused_data import (
    get_cleanup_preview,
    get_data_usage_statistics
)

# Preview cleanup counts
counts = await get_cleanup_preview(days_threshold=30)
print(counts)  # {'document_chunks': 100, 'entities': 50, ...}

# Get detailed statistics
stats = await get_data_usage_statistics()
for table_name, table_stats in stats.items():
    print(f"{table_name}:")
    print(f"  Total: {table_stats['total_records']}")
    print(f"  Never accessed: {table_stats['never_accessed']}")
    print(f"  Recently accessed: {table_stats['recently_accessed']}")
    print(f"  Access rate: {table_stats['accessed_percentage']}%")

# Statistics for custom graph
stats = await get_data_usage_statistics(schema='my_custom_graph')
```

## Core Functions

### Cleanup Module (`cleanup_unused_data.py`)

#### `discover_tracked_tables(metadata, schema=None)`
Discover all tables that have a `last_accessed` column.

**Parameters:**
- `metadata`: SQLAlchemy MetaData instance
- `schema`: Optional schema name for custom graphs

**Returns:** List of Table objects

#### `delete_unused_data(session, days_threshold=30, schema=None, dry_run=False)`
Delete unused data from all tracked tables.

**Parameters:**
- `session`: Database session
- `days_threshold`: Days without access to trigger deletion
- `schema`: Optional schema name
- `dry_run`: If True, only count without deleting

**Returns:** Dictionary mapping table names to deletion counts

#### `get_unused_data_counts(session, days_threshold=30, schema=None)`
Get count of unused records per table without deleting.

#### `get_table_statistics(session, schema=None)`
Get detailed statistics for all tracked tables.

### Access Tracking Module (`access_tracking.py`)

#### `update_last_accessed(session, entity_ids, table_name, schema=None)`
Update last_accessed timestamp for specified entities.

**Parameters:**
- `session`: Database session
- `entity_ids`: Single UUID or list of UUIDs
- `table_name`: Name of the table
- `schema`: Optional schema name

#### `mark_entity_accessed(session, entity_id, table_name, schema=None)`
Convenience function to mark a single entity as accessed.

**Returns:** Boolean indicating success

#### `bulk_update_last_accessed(session, updates, schema=None)`
Perform bulk updates across multiple tables.

**Parameters:**
- `updates`: List of dicts with 'table_name' and 'entity_ids'

## Custom Graph Support

### Creating Custom Graphs with Access Tracking

To enable access tracking for your custom graph:

1. **Add `last_accessed` column** to your tables:

```python
from sqlalchemy import Column, DateTime
from datetime import datetime, timezone

class CustomNode(Base):
    __tablename__ = 'custom_nodes'
    __table_args__ = {'schema': 'my_custom_graph'}
    
    id = Column(UUID, primary_key=True)
    # ... your other columns ...
    last_accessed = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

2. **Run migrations** to create the column:

```python
# In your migration file
def upgrade():
    op.add_column(
        'custom_nodes',
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        schema='my_custom_graph'
    )
```

3. **Use the cleanup and tracking functions** with your schema:

```python
# Cleanup
await cleanup_unused_data(
    days_threshold=30,
    schema='my_custom_graph'
)

# Track access
await mark_entity_accessed(
    session=session,
    entity_id=node_id,
    table_name='custom_nodes',
    schema='my_custom_graph'
)
```

## Testing

Comprehensive test coverage includes:

- Default graph cleanup and tracking
- Custom graph schema support
- Bulk operations
- Error handling
- Dry-run functionality
- Statistics generation

Run tests:

```bash
pytest cognee/tests/unit/test_cleanup_unused_data.py -v
pytest cognee/tests/unit/test_access_tracking.py -v
```

## Migration

The required `last_accessed` column is added via migration:

```
cognee/infrastructure/databases/relational/sqlalchemy/migrations/versions/xxxx_add_last_accessed_columns.py
```

This migration adds the column to all default graph tables:
- `document_chunks`
- `entities`
- `summaries`
- `associations`
- `metadata`

## Performance Considerations

- **Indexes**: Consider adding indexes on `last_accessed` columns for large datasets
- **Batch Processing**: Bulk operations are more efficient than individual updates
- **Dry Runs**: Always test with `dry_run=True` first
- **Off-Peak Hours**: Schedule cleanup during low-traffic periods

## Error Handling

All functions include comprehensive error handling:

- Exceptions are logged with full traceback
- Database transactions are rolled back on error
- Operations continue even if individual tables fail
- Clear error messages in `CleanupResult.errors`

## Logging

The module uses Python's standard logging framework:

```python
import logging

# Enable debug logging
logging.getLogger('cognee.modules.data').setLevel(logging.DEBUG)
```

Log levels:
- `DEBUG`: Table discovery, individual operations
- `INFO`: Cleanup results, statistics
- `ERROR`: Failures, exceptions

## Related Issues

- [#1335](https://github.com/topoteretes/cognee/issues/1335) - Original feature request

## Contributing

When extending this module:

1. Maintain schema-agnostic design
2. Add comprehensive tests
3. Update this documentation
4. Follow existing error handling patterns
5. Use type hints

## License

Same as Cognee project license.
