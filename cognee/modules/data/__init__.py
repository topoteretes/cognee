"""Data management module for Cognee.

This module provides utilities for data cleanup and access tracking
across all graph databases, including both default and custom graphs.
"""

from cognee.modules.data.cleanup_unused_data import (
    discover_tracked_tables,
    delete_unused_data,
    get_unused_data_counts,
    get_table_statistics,
)

from cognee.modules.data.access_tracking import (
    get_tracked_tables,
    update_last_accessed,
    bulk_update_last_accessed,
    mark_entity_accessed,
)

__all__ = [
    # Cleanup functions
    'discover_tracked_tables',
    'delete_unused_data',
    'get_unused_data_counts',
    'get_table_statistics',
    # Access tracking functions
    'get_tracked_tables',
    'update_last_accessed',
    'bulk_update_last_accessed',
    'mark_entity_accessed',
]
