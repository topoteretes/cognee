# Zendesk Tickets Connector

A Zendesk data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn your customer support ticket history and conversation logs into searchable long-term memory.

It exposes a `dlt` source you pass directly to `cognee.remember(...)`, integrating seamlessly into Cognee's incremental graph and vector pipelines.

---

## Features

- **Incremental Sync**: Uses Zendesk's Incremental Export API with epoch timestamps (`start_time`), ensuring repeat runs only sync modified tickets.
- **Full Thread Preservation**: Comments, internal notes, and agent responses are folded into each ticket's content document to maintain conversational context.
- **Tombstone & Deletion Handling**: Tracks ticket status transitions to `deleted` and emits `_deleted: True` tombstones to keep the memory graph clean.
- **Deterministic Node Sets**: Emits `cognee_node_set: zendesk:<subdomain>:tickets` for scoped recalls.

---

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import zendesk_source

await cognee.remember(
    zendesk_source(
        subdomain="your-company",
        email="agent@your-company.com",
        api_token="<zendesk-api-token>",
    ),
    dataset_name="zendesk_support",
    primary_key="id",
    write_disposition="merge",  # REQUIRED: prevents dropping tables on re-sync
    max_rows_per_table=0,
)
```

Alternatively, you can authenticate via OAuth bearer token:

```python
await cognee.remember(
    zendesk_source(
        subdomain="your-company",
        oauth_token="<oauth-access-token>",
    ),
    dataset_name="zendesk_support",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

---

## Data Schema

Each ingested row contains:
- `id`: Unique identifier (`ticket:<ticket_id>`)
- `title`: Ticket subject or default label
- `content`: Formatted markdown including status, priority, description, and comment history
- `url`: Direct link to ticket in Zendesk Agent Workspace
- `_deleted`: Boolean flag indicating deleted tickets
- `cognee_node_set`: Scoped graph domain identifier (`zendesk:<subdomain>:tickets`)
- `document_source`: Set to `"zendesk"`
