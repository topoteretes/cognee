# Jira Issues Connector (`jira_source`)

Ingests issues, descriptions (including Atlassian Document Format / ADF), subtasks, and comments from selected Atlassian Jira Cloud projects as incremental, deletion-aware document rows for `cognee.remember(...)`.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import jira_source

await cognee.remember(
    jira_source(
        base_url="https://acme.atlassian.net",
        access_token="user@acme.com:<jira_api_token>",  # or OAuth 2.0 Bearer token
        project_keys=["ENG", "PLAT"],
    ),
    dataset_name="jira",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

## Features

- **Document-mode DLT Resource**: Tagged with `cognee_document_source = "jira"` and `write_disposition = "merge"`.
- **Per-Project Node Sets**: Assigns `cognee_node_set = ["jira:<PROJECT_KEY>"]` so queries can be scoped to specific Jira projects.
- **ADF & Plain-Text Rendering**: Deterministically converts Atlassian Document Format (ADF) descriptions and comments into structured text.
- **Incremental & Tombstone Sync**: Tracks per-project `updated` cursors and emits `_deleted = True` tombstones when a project is deselected or an issue is removed during full reconciliation.
- **Bounded & Resumable**: Stops cleanly on HTTP 429 rate limits or `max_requests` budget limits while recording telemetry in `resource.cognee_sync_stats`.
