# Zendesk Tickets Connector (`zendesk_source`)

Ingests support tickets, status, priority, problem descriptions, tags, and conversation audit comments from Zendesk as incremental, deletion-aware document rows for `cognee.remember(...)`.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import zendesk_source

await cognee.remember(
    zendesk_source(
        subdomain="acme",
        access_token="agent@acme.com/token:<zendesk_api_token>",  # or OAuth 2.0 Bearer token
        query="type:ticket status<solved",  # optional filter query
    ),
    dataset_name="zendesk",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

## Features

- **Document-mode DLT Resource**: Tagged with `cognee_document_source = "zendesk"` and `write_disposition = "merge"`.
- **Per-Subdomain Node Sets**: Assigns `cognee_node_set = ["zendesk:<subdomain>"]` so queries can be scoped to specific Zendesk brand/workspace instances.
- **Structured Conversation Folding**: Deterministically folds ticket problem descriptions, metadata, and audit conversation threads into a unified memory document.
- **Incremental & Cursor-Aware**: Supports cursor-based incremental sync (`after_cursor`) for export and search APIs, tracking cursors and `known_ids` in `dlt.current.resource_state()`.
- **Bounded & Resumable**: Stops cleanly on HTTP 429 rate limits or `max_requests` budget limits while recording telemetry in `resource.cognee_sync_stats`.
