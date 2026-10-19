# Teable No-Code Database Data-Source Connector

A Teable data-source connector for [cognee](https://github.com/topoteretes/cognee): sync structured tables, relational link fields, and records from Teable (the open-source Airtable alternative built on Postgres) into AI memory incrementally, with automatic forget-on-deletion.

Exposes a `dlt` resource intended for `cognee.remember(...)`, integrating directly into Cognee's document-mode ingestion pipeline. Records flow through chunking and LLM knowledge-graph extraction, enabling AI agents to reason over team CRMs, content calendars, research trackers, and inventories.

## Features

- **Multi-Table & Record Ingestion**: Ingests field-name keyed records with values across all column types (text, numbers, booleans, formulas, rollups).
- **Cross-Table Relational Link Fields**: Resolves link cells (e.g. `Contacts -> Companies`) as semantic references, allowing Cognee to construct cross-table knowledge graph edges.
- **Content-Hash Change Detection**:
  - Computes SHA-256 hashes of rendered record markdown.
  - Catches changes that traditional `lastModifiedTime` cursors miss (such as renamed linked records, renamed column headers, or recalculated formula fields).
  - Skips re-cognifying unchanged records, saving downstream LLM costs.
- **Forget-on-Delete (Tombstones)**:
  - Records removed from complete table listings (or detected via Teable's `/trash/items` API) are emitted as `_deleted=True` tombstones under `write_disposition="merge"`, enabling Cognee's orphan cleanup to remove purged entities.
- **Self-Hosted & Cloud Compatibility**:
  - Works with Teable Cloud (`https://app.teable.ai/api`) and self-hosted Docker/Kubernetes instances.
  - Zero external vendor SDK dependencies (uses `httpx`, a core Cognee dependency).
  - Personal Access Tokens are masked in strings and logs (`teab...1234`).
- **Privacy Defaults**:
  - User email addresses and names in trash/audit metadata (`userMap`) are excluded from ingested documents.

## Setup & Credentials

1. In your Teable workspace (cloud or self-hosted), navigate to **User Settings** → **Personal Access Tokens**.
2. Create a new token with read permissions on your target spaces/bases.
3. Set your environment variables:

```bash
export TEABLE_API_KEY="teable_pat_your_token_here"
export TEABLE_BASE_URL="https://app.teable.ai/api"  # or your self-hosted URL
```

## Quickstart

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import teable_source

async def main():
    # Ingest CRM and contacts tables into a dedicated dataset
    await cognee.remember(
        teable_source(
            api_key="teable_pat_your_token_here",
            base_url="https://app.teable.ai/api",
            table_ids=["tblCompanies123", "tblContacts456"],
            table_names={
                "tblCompanies123": "Companies",
                "tblContacts456": "Contacts",
            },
        ),
        dataset_name="crm_memory",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Query your structured business memory with graph completion
    answer = await cognee.search(
        query_text="Which enterprise clients asked about custom SLAs, and who is our primary contact there?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=["crm_memory"],
    )
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | Teable Personal Access Token (defaults to `TEABLE_API_KEY`). |
| `base_url` | `str` | `"https://app.teable.ai/api"` | Teable API endpoint URL. |
| `table_ids` | `Sequence[str]` | **Required** | List of Teable table IDs to ingest. |
| `table_names` | `dict[str, str] \| None` | `None` | Optional mapping of table IDs to human-readable names. |
| `resource_name` | `str \| None` | `"teable_<table_id>"` | Staging table identifier. |
| `page_size` | `int` | `100` | Pagination page size for Teable REST API calls. |
| `service` | `TeableClient \| None` | `None` | Pre-configured client instance for testing and injection. |

## Document Schemas & Graph Entities

- **Document ID**: `teable:{table_id}:{record_id}`
- **Title**: `<table_name>: <primary_record_name>`
- **Content**: Structured markdown containing properties, field values, and linked cross-table relations (`[Target Name](teable:<target_id>)`).
- **Tombstones**: When a record is deleted, `{"id": "teable:<table_id>:<record_id>", "_deleted": True}` is yielded to trigger Cognee orphan node removal.
