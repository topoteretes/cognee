# Zotero connector

`cognee-community-connector-zotero` exposes a Zotero library as a `dlt` resource
that can be passed to `cognee.remember`.

## Setup

1. Create a Zotero API key with read access to the target user or group library.
2. Export `ZOTERO_API_KEY` and either `ZOTERO_USER_ID` or `ZOTERO_GROUP_ID`.
3. Install this package alongside `cognee[dlt]`.

## Example

```python
import cognee
from cognee_community_connector_zotero import zotero_source

await cognee.remember(
    zotero_source(
        api_key="zotero-api-key",
        user_id="123456",
        include_references=True,
        include_notes=True,
        include_attachments=True,
    ),
    dataset_name="zotero_library",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)

answer = await cognee.search("Which references discuss knowledge graphs?", datasets=["zotero_library"])
```

The connector stores the Zotero `Last-Modified-Version` cursor in DLT source
state and sends it back as `since` on later runs. Deleted Zotero item keys are
emitted as `_deleted` tombstones, using DLT's `hard_delete` column so cognee's
shared DLT orphan cleanup removes stale graph/vector records on the next sync.
