# HubSpot CRM Connector (`hubspot_source`)

Ingests contacts, companies, deals, and engagement notes from HubSpot CRM as incremental, deletion-aware document rows for `cognee.remember(...)`.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import hubspot_source

await cognee.remember(
    hubspot_source(
        access_token="pat-na1-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Private App Access Token
        portal_id="12345678",  # optional: automatically resolved if omitted
        include_contacts=True,
        include_companies=True,
        include_deals=True,
        include_notes=True,
    ),
    dataset_name="hubspot",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

## Features

- **Document-mode DLT Resource**: Tagged with `cognee_document_source = "hubspot"` and `write_disposition = "merge"`.
- **Per-Portal Node Sets**: Assigns `cognee_node_set = ["hubspot:<portal_id>:<kind>", "hubspot:<portal_id>"]` so queries can be scoped to specific HubSpot portals or object domains (`contacts`, `companies`, `deals`, `notes`).
- **Structured Content Rendering**: Flattens contacts, company attributes, deal stages/amounts, and engagement note bodies (with HTML markup stripped) into clean text representations for vector embedding and knowledge graph creation.
- **Incremental & Cursor-Aware**: Tracks cursor timestamps (`hs_lastmodifieddate`) in `dlt.current.resource_state()`, fetching only modified or new entities on subsequent runs.
- **Resilient & Bounded Execution**: Gracefully catches HTTP 429 rate limits or request budget caps, recording telemetry in `resource.cognee_sync_stats`.
