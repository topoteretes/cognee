# Linkding SDK connector

A Linkding bookmark data-source connector for [cognee](https://github.com/topoteretes/cognee): sync bookmarks, notes, and tags from a self-hosted [Linkding](https://github.com/sissbruecker/linkding) instance into Cognee memory.

It exposes a `dlt` resource you pass to `cognee.remember(...)`. Each bookmark becomes a document with title, URL, description, Markdown notes, tags, and status.

## Features

- **Self-Hosted Bookmark Sync**: Ingests bookmarks from any local or remote Linkding instance via REST API token authentication.
- **Notes & Metadata**: Ingests user notes, descriptions, unread flags, and archive status into structured document content.
- **Scoped Node Sets**: Attaches `linkding:bookmark` and `linkding:tag:<tag>` node sets to enable precise graph filtering and search retrieval.
- **Incremental Synchronization**: Respects `modified_since` cursor to fetch only bookmarks created or edited after the previous sync.
- **Forget-on-Delete**: Detects deleted bookmarks and emits `_deleted: True` tombstone records to prune removed bookmarks from Cognee.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import linkding_source

source = linkding_source(
    base_url="http://localhost:9090",
    api_token="your_linkding_api_token",
    include_archived=True,
)

await cognee.remember(
    source,
    dataset_name="bookmarks",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
    self_improvement=False,
)

assert source.cognee_sync_stats["synced"] >= 0
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `base_url` | Base URL of the Linkding instance | required |
| `api_token` | REST API token from Linkding Settings -> Integrations | required |
| `resource_name` | Identifier for dlt resource state | `linkding_<sha256_hash>` |
| `include_archived` | Ingest archived bookmarks in addition to active bookmarks | `False` |
| `bundle` | Filter bookmarks by Linkding bundle ID | `None` |
| `verify_ssl` | Verify TLS certificates (set `False` for self-signed certs) | `True` |
| `check_active` | Optional host authorization checkpoint callback | `None` |
