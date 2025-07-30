# User Data Sharing Pipeline Implementation

## Overview

This implementation adds a User Data Sharing Pipeline that enables transferring data from one user's Kuzu + LanceDB databases to another user's Kuzu + LanceDB databases with proper metastore updates for ownership and permissions.

## Features Implemented

### 1. Data Export Endpoint
- **Endpoint**: `GET /v1/data/export/{dataset_id}`
- **Purpose**: Export all data associated with a dataset from source user's databases
- **Authentication**: Requires user authentication and "share" permission on dataset
- **Exports**:
  - Graph nodes and edges from Kuzu database
  - Vector embeddings from LanceDB collections
  - Dataset metadata from PostgreSQL metastore
  - User permissions and access control lists

### 2. Data Import Endpoint
- **Endpoint**: `POST /v1/data/import`
- **Purpose**: Import data transfer bundle into target user's databases
- **Authentication**: Requires user authentication
- **Features**:
  - Accepts JSON transfer bundle as multipart file upload
  - Updates ownership from source_user_id to target_user_id
  - Preserves data integrity and relationships
  - Creates new dataset with target user ownership
  - Updates PostgreSQL metastore with new permissions

### 3. Transfer Bundle Format
The transfer bundle is a JSON structure containing:
```json
{
  "dataset_id": "original-dataset-uuid",
  "metadata": {
    "dataset": { /* dataset info */ },
    "data_items": [ /* data items */ ]
  },
  "graph_data": {
    "nodes": [ /* kuzu nodes */ ],
    "edges": [ /* kuzu edges */ ],
    "node_count": 10,
    "edge_count": 15
  },
  "vector_data": {
    "collections": { /* lancedb collections */ },
    "collection_count": 1
  },
  "metastore_data": {
    "permissions": [ /* ACL permissions */ ],
    "dataset_database_config": { /* database config */ }
  },
  "created_at": "2023-01-01T00:00:00",
  "source_user_id": "source-user-uuid",
  "version": "1.0"
}
```

## File Structure

```
cognee/api/v1/data/
├── __init__.py
└── routers/
    ├── __init__.py
    └── get_data_router.py

cognee/modules/data/methods/
├── export_dataset_data.py
├── import_dataset_data.py
└── __init__.py (updated)

cognee/tests/
└── test_data_sharing.py
```

## Implementation Details

### Export Process
1. **Permission Check**: Verify user has "share" permission on dataset
2. **Metadata Export**: Extract dataset and data items from PostgreSQL
3. **Graph Export**: Extract nodes and edges from Kuzu database with dataset filtering
4. **Vector Export**: Extract embeddings from LanceDB collections
5. **Metastore Export**: Extract permissions and database configurations
6. **Bundle Creation**: Package all data into transfer format

### Import Process
1. **Validation**: Validate transfer bundle structure and format
2. **Dataset Creation**: Create new dataset with target user ownership
3. **ID Mapping**: Generate new UUIDs for all imported entities
4. **Graph Import**: Import nodes and edges with updated ownership
5. **Vector Import**: Import embeddings into target collections
6. **Metastore Update**: Update permissions and database configs
7. **Data Items**: Create new data items with target user ownership

### Key Features

#### User Mapping
- Automatically updates all `user_id` and `owner_id` references
- Maps source user permissions to target user
- Preserves data relationships while changing ownership

#### Data Integrity
- Maintains graph relationships with new node IDs
- Preserves vector embeddings and their associations
- Ensures dataset consistency across databases

#### Permission Management
- Target user gets full permissions (read, write, share, delete)
- Original source permissions are not modified
- ACL entries are created for the new dataset

#### Error Handling
- Comprehensive validation of transfer bundles
- Graceful error handling with rollback capabilities
- Detailed error messages for debugging

## API Usage Examples

### Export Dataset
```bash
curl -X GET "http://localhost:8000/api/v1/data/export/{dataset_id}" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json"
```

### Import Dataset
```bash
curl -X POST "http://localhost:8000/api/v1/data/import" \
  -H "Authorization: Bearer {token}" \
  -F "transfer_file=@transfer_bundle.json" \
  -F 'import_request={"target_dataset_name": "imported_dataset", "preserve_relationships": true}'
```

## Testing

The implementation includes comprehensive tests covering:
- Successful export/import scenarios
- Permission validation
- Error handling for invalid files
- Integration testing of the complete pipeline

Run tests with:
```bash
python -m pytest cognee/tests/test_data_sharing.py -v
```

## Security Considerations

1. **Authentication**: All endpoints require valid user authentication
2. **Authorization**: Export requires "share" permission on source dataset
3. **Data Validation**: Transfer bundles are validated before processing
4. **Ownership Transfer**: Ensures clean ownership transfer without data leaks
5. **Audit Trail**: All operations are logged for security tracking

## Future Enhancements

1. **Partial Imports**: Support for importing specific portions of datasets
2. **Incremental Sync**: Support for syncing changes between shared datasets
3. **Compression**: Compress transfer bundles for large datasets
4. **Async Processing**: Handle large dataset transfers asynchronously
5. **Cross-Instance Sharing**: Support sharing between different Cognee instances

## Configuration

The feature uses existing Cognee configuration for:
- Database connections (Kuzu, LanceDB, PostgreSQL)
- User authentication and permissions
- File storage for transfer bundles

No additional configuration is required.

## Dependencies

This implementation leverages existing Cognee infrastructure:
- `cognee.infrastructure.databases.graph` - Kuzu database operations
- `cognee.infrastructure.databases.vector` - LanceDB operations
- `cognee.infrastructure.databases.relational` - PostgreSQL operations
- `cognee.modules.users.permissions` - Permission management
- `cognee.modules.data.methods` - Dataset management

## Troubleshooting

Common issues and solutions:

1. **Permission Denied**: Ensure user has "share" permission on source dataset
2. **Invalid Transfer File**: Verify JSON format and required fields
3. **Import Failures**: Check database connectivity and target user permissions
4. **Large Datasets**: Consider implementing chunked transfer for very large datasets

## Contributing

When extending this feature:
1. Follow existing code patterns and error handling
2. Add comprehensive tests for new functionality
3. Update this documentation
4. Ensure backward compatibility with existing transfer bundle format
