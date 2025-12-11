# Custom Data Labels Feature

## Overview

The Custom Data Labels feature allows users to provide human-friendly names for data items when adding them to Cognee. This is especially useful for text data where the default name is just the content hash, making it difficult to identify what the data represents at a glance.

## Problem Statement

When adding raw text or other unstructured data to Cognee, the system assigns a default name based on the content hash. This makes it hard for users to:
- Identify what specific data items represent
- Organize and manage data effectively
- Track data lineage and purpose
- Quickly understand dataset contents

## Solution

The `DataItem` dataclass allows you to attach a custom label to any data being ingested. This label is then stored in the database and returned with the data information.

## Usage

### Basic Usage

```python
from cognee import DataItem
import cognee

# Create a DataItem with custom label
data_item = DataItem(
    data="Your actual data content here",
    label="My Custom Label"
)

# Add it to Cognee
await cognee.add(
    data=data_item,
    dataset_name="my_dataset",
    user=user
)
```

### Multiple Items with Labels

```python
from cognee import DataItem

items = [
    DataItem(
        data="Q3 financial results showing 25% revenue growth",
        label="Q3 Financial Report"
    ),
    DataItem(
        data="Customer satisfaction metrics and analysis",
        label="Customer Satisfaction Report"
    ),
]

await cognee.add(
    data=items,
    dataset_name="reports",
    user=user
)
```

### Mixed Data (with and without labels)

```python
from cognee import DataItem

mixed_data = [
    DataItem(data="Important report", label="Annual Report 2024"),
    "Regular text data without custom label",  # No label
    DataItem(data="Another report", label="Quarterly Review"),
]

await cognee.add(
    data=mixed_data,
    dataset_name="reports",
    user=user
)
```

### Retrieving Data with Labels

```python
from cognee.modules.data.methods import get_dataset_data, get_datasets

# Get all datasets
datasets = await get_datasets(user.id)

# Get a specific dataset
my_dataset = next((d for d in datasets if d.name == "my_dataset"), None)

# Get all data items in the dataset
dataset_data = await get_dataset_data(my_dataset.id)

# Access labels
for data_item in dataset_data:
    if data_item.label:
        print(f"Label: {data_item.label}")
    else:
        print("No label provided")
```

### API Response

When retrieving data via the REST API, the label is included in the response:

```json
{
  "id": "12345678-1234-5678-1234-567812345678",
  "name": "text_abc123def456.txt",
  "label": "Q3 Financial Report",
  "extension": "txt",
  "mimeType": "text/plain",
  "rawDataLocation": "/path/to/data",
  "createdAt": "2024-12-11T10:00:00Z",
  "updatedAt": "2024-12-11T10:00:00Z"
}
```

## Implementation Details

### DataItem Dataclass

Located in `cognee/modules/data/models/DataItem.py`:

```python
@dataclass
class DataItem:
    """
    A dataclass for providing data with optional custom labels.
    
    Attributes:
        data: The actual data to be ingested (str, file, DataPoint, etc.)
        label: Optional custom label for human-friendly identification
    """
    data: Any
    label: Optional[str] = None
```

### Database Schema

The `Data` model now includes a `label` column:

```python
label = Column(String, nullable=True)  # Custom label for user-friendly identification
```

### Processing Flow

1. User creates a `DataItem` with data and optional label
2. During `ingest_data`, the system detects if the item is a `DataItem`
3. If it is, the label is extracted and stored separately from the data
4. The data value is processed normally through the ingestion pipeline
5. The label is persisted in the `Data.label` column
6. When retrieving data, the label is included in responses

## Database Migration

A migration file (`alembic/versions/f5a1b2c3d4e5_add_label_column_to_data.py`) has been created to add the `label` column to existing databases.

To apply the migration:

```bash
# Using alembic directly
alembic upgrade head

# Or if using the project's migration system
uv run python -m alembic upgrade head
```

## Backward Compatibility

- The `label` column is nullable, so existing data without labels will work fine
- Plain data (without `DataItem` wrapper) continues to work as before
- The label is optional - if not provided, it defaults to `None`
- All existing APIs and functionality remain unchanged

## API Changes

### New Exports

- `DataItem` is now exported from the main `cognee` module

### Modified DTOs

- `DataDTO` in the datasets router now includes the optional `label` field

### Affected Endpoints

- `GET /v1/datasets/{dataset_id}/data` - Now includes label in response
- `GET /v1/datasets/{dataset_id}/data/{data_id}/raw` - Unchanged (still returns raw data)

## Testing

Comprehensive tests have been added in `cognee/tests/test_data_item_label.py`:

- Test with custom labels
- Test without labels
- Test plain string data (backward compatibility)
- Test multiple items with different labels
- Test API response includes label

Run tests with:

```bash
uv run pytest cognee/tests/test_data_item_label.py -v
```

## Examples

A complete example is provided in `examples/python/data_item_custom_labels_example.py`:

```bash
uv run python examples/python/data_item_custom_labels_example.py
```

## Use Cases

1. **Document Management**: Label documents by their type or purpose
   ```python
   DataItem(data=doc_content, label="Contract - Q4 2024")
   ```

2. **Multi-Source Data**: Identify the source of ingested data
   ```python
   DataItem(data=email_content, label="Email from CEO (Dec 11)")
   ```

3. **Dataset Organization**: Group related items with meaningful names
   ```python
   items = [
       DataItem(data=q1_data, label="Q1 Financial Summary"),
       DataItem(data=q2_data, label="Q2 Financial Summary"),
       DataItem(data=q3_data, label="Q3 Financial Summary"),
   ]
   ```

4. **Audit Trails**: Track data for compliance
   ```python
   DataItem(data=report, label="GDPR Compliance Report - 2024")
   ```

## Future Enhancements

- Support for label-based filtering/searching
- Label suggestions based on content analysis
- Bulk label updates
- Label analytics and insights
