# Data Management Module

This module provides utilities for managing data lifecycle in Cognee, including automatic cleanup of unused Data entries and access tracking at the Data model level.

## Overview

The data module works at the **Data model level** to ensure proper cleanup of related graph and vector database entries. It consists of two main components:

1. **Cleanup Unused Data** (`cleanup_unused_data.py`) - Data-level cleanup based on access patterns
2. **Access Tracking** (`access_tracking.py`) - Track when Data entries are accessed via reference table

## Key Features

✅ **Data-Level Operations**: Works at Data model level, not entity-level tables
✅ **Reference Table Tracking**: Uses `data_access_tracking` table to avoid frequent writes on main Data table
✅ **Proper Cascade Deletion**: Leverages existing deletion infrastructure to clean up graph and vector DB entries
✅ **Safe Operations**: Supports dry-run mode to preview deletions
✅ **Comprehensive Statistics**: Get detailed usage statistics for Data entries
✅ **Efficient Bulk Operations**: PostgreSQL upsert for high-performance tracking

## Architecture

### Data-Level Design

The module operates at the Data model level rather than tracking individual entity tables. This design provides several benefits:

- **Proper Cascade Deletion**: When Data is deleted, related graph and vector database entries are automatically cleaned up through existing deletion infrastructure
- **Simpler Architecture**: One reference table instead of columns in multiple tables
- **Better Performance**: Reference table approach avoids frequent writes on main Data table
- **Unified Interface**: All data types are tracked through a single mechanism

### Database Schema

#### data_access_tracking Table

```sql
CREATE TABLE data_access_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_id UUID NOT NULL REFERENCES data(id) ON DELETE CASCADE,
    last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(data_id)
);

CREATE INDEX ix_data_access_tracking_data_id ON data_access_tracking(data_id);
CREATE INDEX ix_data_access_tracking_last_accessed ON data_access_tracking(last_accessed);
```

### How It Works

1. **Access Tracking**: When Data is accessed, `track_data_access()` updates or inserts a record in `data_access_tracking`
2. **Cleanup Queries**: `get_unused_data_ids()` finds Data entries with old `last_accessed` timestamps or never tracked
3. **Data Deletion**: `cleanup_unused_data()` uses existing `delete_data_by_id()` function for proper cascade deletion
4. **Graph/Vector Cleanup**: Existing deletion infrastructure ensures related graph nodes and vector embeddings are removed

## Usage

### Access Tracking

#### Track Single Data Access

```python
from cognee.modules.data.access_tracking import track_data_access
from uuid import UUID

# Track when a Data entry is accessed
data_id = UUID('...')
await track_data_access(session, data_id)
```

#### Bulk Track Data Access

```python
from cognee.modules.data.access_tracking import bulk_track_data_access

# Efficiently track multiple Data entries
data_ids = [UUID('...'), UUID('...'), UUID('...')]
count = await bulk_track_data_access(session, data_ids)
print(f"Tracked {count} Data entries")
```

#### Get Access Information

```python
from cognee.modules.data.access_tracking import (
    get_data_last_accessed,
    get_data_access_count
)

# Get last accessed timestamp
last_access = await get_data_last_accessed(session, data_id)
print(f"Last accessed: {last_access}")

# Get access count
count = await get_data_access_count(session, data_id)
print(f"Accessed {count} times")
```

### Data Cleanup

#### Preview Cleanup (Dry Run)

```python
from cognee.modules.data.cleanup_unused_data import cleanup_unused_data

# Preview what would be deleted (dry run)
result = await cleanup_unused_data(
    session=session,
    days_threshold=30,
    dry_run=True
)

print(f"Would delete {result['deleted_count']} Data entries")
print(f"Data IDs: {result['unused_data_ids']}")
```

#### Perform Actual Cleanup

```python
# Actually delete unused Data entries
result = await cleanup_unused_data(
    session=session,
    days_threshold=30,
    dry_run=False
)

if result['success']:
    print(f"Successfully deleted {result['deleted_count']} Data entries")
else:
    print(f"Cleanup failed: {result['errors']}")
```

#### Get Cleanup Statistics

```python
from cognee.modules.data.cleanup_unused_data import get_cleanup_statistics

# Get statistics about Data usage
stats = await get_cleanup_statistics(session, days_threshold=30)

print(f"Total Data entries: {stats['total_data_count']}")
print(f"Tracked entries: {stats['tracked_count']}")
print(f"Never accessed: {stats['untracked_count']}")
print(f"Unused (>30 days): {stats['unused_count']}")
print(f"Active entries: {stats['active_count']}")
```

### Task Integration

The `cognee/tasks/cleanup_unused_data.py` module provides a high-level task interface:

```python
from cognee.tasks.cleanup_unused_data import (
    cleanup_unused_data,
    get_data_usage_statistics
)

# Use task function (handles session automatically)
result = await cleanup_unused_data(
    days_threshold=30,
    dry_run=True
)

# Get usage statistics
stats = await get_data_usage_statistics(days_threshold=30)
```

## Migration

The module includes an Alembic migration to create the `data_access_tracking` table:

```
alembic/versions/2f3a4b5c6d7e_add_data_access_tracking_table.py
```

Run migrations to set up the table:

```bash
alembic upgrade head
```

## API Reference

### access_tracking.py

#### `track_data_access(session, data_id) -> bool`

Track access to a Data entry.

- **Args**:
  - `session`: AsyncSession - Database session
  - `data_id`: UUID - ID of the Data entry
- **Returns**: bool - True if successful

#### `bulk_track_data_access(session, data_ids) -> int`

Track access to multiple Data entries efficiently.

- **Args**:
  - `session`: AsyncSession - Database session
  - `data_ids`: List[UUID] - List of Data IDs
- **Returns**: int - Number of successfully tracked entries

#### `get_data_last_accessed(session, data_id) -> Optional[datetime]`

Get the last accessed timestamp for a Data entry.

- **Args**:
  - `session`: AsyncSession - Database session
  - `data_id`: UUID - ID of the Data entry
- **Returns**: Optional[datetime] - Last accessed time or None

#### `get_data_access_count(session, data_id) -> int`

Get the access count for a Data entry.

- **Args**:
  - `session`: AsyncSession - Database session
  - `data_id`: UUID - ID of the Data entry
- **Returns**: int - Number of times accessed (0 if never)

### cleanup_unused_data.py

#### `get_unused_data_ids(session, days_threshold) -> List[str]`

Get IDs of Data entries that haven't been accessed within threshold.

- **Args**:
  - `session`: AsyncSession - Database session
  - `days_threshold`: int - Number of days to consider as unused
- **Returns**: List[str] - List of Data UUIDs as strings

#### `cleanup_unused_data(session, days_threshold, dry_run) -> Dict`

Clean up Data entries that haven't been accessed within threshold.

- **Args**:
  - `session`: AsyncSession - Database session
  - `days_threshold`: int - Number of days (default: 30)
  - `dry_run`: bool - Preview only if True (default: True)
- **Returns**: Dict with keys:
  - `success`: bool
  - `dry_run`: bool
  - `deleted_count`: int
  - `unused_data_ids`: List[str]
  - `errors`: List[str]
  - `timestamp`: datetime

#### `get_cleanup_statistics(session, days_threshold) -> Dict`

Get statistics about Data entries and access patterns.

- **Args**:
  - `session`: AsyncSession - Database session
  - `days_threshold`: int - Threshold for unused (default: 30)
- **Returns**: Dict with keys:
  - `total_data_count`: int
  - `tracked_count`: int
  - `untracked_count`: int
  - `unused_count`: int
  - `active_count`: int

## Performance Considerations

### PostgreSQL Upsert

The module uses PostgreSQL's `INSERT ... ON CONFLICT` for efficient upserts:

```python
# Automatically handles concurrent access without race conditions
await track_data_access(session, data_id)
```

### Batch Processing

Bulk operations process records in batches of 100 to avoid overwhelming the database:

```python
# Efficiently handles thousands of Data IDs
await bulk_track_data_access(session, large_list_of_ids)
```

### Indexes

Two indexes optimize query performance:

- `ix_data_access_tracking_data_id`: Fast lookups by Data ID
- `ix_data_access_tracking_last_accessed`: Efficient cleanup queries

## Integration with Retrievers

When integrating with data retrievers, add access tracking:

```python
from cognee.modules.data.access_tracking import track_data_access

# In your retriever function
async def retrieve_data(data_id):
    # Fetch the data
    data = await fetch_data(data_id)
    
    # Track access
    await track_data_access(session, data_id)
    
    return data
```

## Related Issues

- [#1335](https://github.com/topoteretes/cognee/issues/1335) - Add task that deletes old data not accessed in a while
- [PR #1531](https://github.com/topoteretes/cognee/pull/1531) - Initial implementation and refactoring

## Testing

Comprehensive tests are available in:

- `cognee/tests/unit/test_cleanup_unused_data.py`
- `cognee/tests/unit/test_access_tracking.py`
- `cognee/tests/integration/test_cleanup_integration.py`
- `cognee/tests/performance/test_cleanup_performance.py`

## Future Enhancements

- Scheduled cleanup tasks (daily/weekly cron jobs)
- Configurable cleanup policies per Data type
- Cleanup notifications and reporting
- Dashboard for monitoring data usage
