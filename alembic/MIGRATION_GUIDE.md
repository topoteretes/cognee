# Database Migration Guide: Last Accessed Timestamps

## Overview

This document describes the database migration `2f3a4b5c6d7e_add_last_accessed_timestamps.py` which adds `last_accessed` timestamp columns to support the cleanup_unused_data feature (Issue #1335).

## Migration Details

**Revision ID:** `2f3a4b5c6d7e`  
**Revises:** `1daae0df1866`  
**Date:** 2025-10-11

## What This Migration Does

This migration adds `last_accessed` timestamp columns to the following tables:

1. **document_chunks** - Tracks when document chunks were last accessed
2. **entities** - Tracks when entities were last accessed
3. **summaries** - Tracks when summaries were last accessed
4. **associations** - Tracks when associations were last accessed
5. **metadata** - Tracks when metadata records were last accessed

### Column Specifications

- **Column Name:** `last_accessed`
- **Type:** `DateTime(timezone=True)`
- **Nullable:** `True`
- **Default:** `CURRENT_TIMESTAMP` (set at time of migration)

### Indexes Created

The migration also creates indexes on the `last_accessed` columns for efficient querying:

- `ix_document_chunks_last_accessed`
- `ix_entities_last_accessed`
- `ix_summaries_last_accessed`
- `ix_associations_last_accessed`
- `ix_metadata_last_accessed`

These indexes enable efficient queries when identifying and cleaning up old, unused data.

## Running the Migration

### Upgrade (Apply Migration)

```bash
alembic upgrade head
```

This will add the `last_accessed` columns and indexes to all relevant tables.

### Downgrade (Rollback Migration)

```bash
alembic downgrade -1
```

This will remove the `last_accessed` columns and their associated indexes.

## Impact on Existing Data

- All existing records will have their `last_accessed` timestamp set to `CURRENT_TIMESTAMP` at the time of migration
- The columns are nullable, so no data validation issues should occur
- The indexes will be built automatically during the migration

## Integration with cleanup_unused_data Feature

This migration is part of the larger cleanup_unused_data feature (Issue #1335). The `last_accessed` timestamps enable:

1. **Data Lifecycle Management** - Track when data was last used
2. **Automated Cleanup** - Identify and remove stale data based on access patterns
3. **Performance Optimization** - Reduce database size by removing unused records
4. **Storage Efficiency** - Reclaim storage space from inactive data

## Application Code Changes Required

After running this migration, application code should be updated to:

1. Update `last_accessed` timestamps when records are queried or used
2. Use the `last_accessed` field in cleanup operations
3. Respect the indexes when querying by `last_accessed`

## Monitoring

After migration, monitor:

1. Index performance and usage
2. Query execution times for cleanup operations
3. Database size and growth patterns
4. Application performance impact

## Rollback Considerations

If you need to roll back this migration:

1. The `last_accessed` data will be lost
2. Any cleanup operations relying on these columns will fail
3. Application code may need to be reverted if it depends on these columns

## Related Files

- Migration file: `alembic/versions/2f3a4b5c6d7e_add_last_accessed_timestamps.py`
- Cleanup implementation: `cognee/tasks/cleanup_unused_data.py`
- Issue: #1335

## Questions or Issues

If you encounter any issues with this migration, please:

1. Check the alembic migration logs
2. Verify database connectivity and permissions
3. Report issues on GitHub with the #1335 tag
4. Review the migration file for any database-specific modifications needed
