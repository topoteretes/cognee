# Figma SDK connector

A Figma data-source connector for [cognee](https://github.com/topoteretes/cognee): turn Figma files,
pages, text elements, reusable components, and comments into knowledge graph memory.

It exposes a `dlt` document source you pass directly to `cognee.remember(...)`, integrating seamlessly
with cognee's existing DLT ingestion subsystem.

## Requirements

Bundled with the Cognee SDK core; no extra package installation is required.

You provide:
- A Figma Personal Access Token (PAT) or OAuth 2.0 access token.
- One or more Figma file keys (found in the Figma URL: `https://www.figma.com/design/<file_key>/<title>`).

## Authentication & Setup

1. Generate a Personal Access Token in your Figma account settings under **Account Settings > Personal access tokens**, or obtain an OAuth 2.0 access token with `file_content:read` and `file_comments:read` scopes.
2. The connector accepts the token directly via `access_token` and never stores it in cleartext or echoes it in exceptions or logs.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import figma_source

await cognee.remember(
    figma_source(
        resource_name="my_figma_designs",
        file_keys=["abcdef1234567890"],
        access_token="<your-figma-token>",
    ),
    dataset_name="figma",
    primary_key="id",
    write_disposition="merge",  # Required for incremental updates and deletions
    max_rows_per_table=0,       # 0 = no row cap, ensuring complete snapshot visibility
)

# Recall design elements scoped to this file
answer = await cognee.recall(
    "What buttons and components are defined in the design?",
    datasets=["figma"],
    node_name=["figma:abcdef1234567890"],
)
```

| Parameter | Type | Meaning | Default |
|---|---|---|---|
| `resource_name` | `str` | Unique name for this connection staging table and state. | Required |
| `file_keys` | `list[str]` | List of Figma file keys to sync. | Required |
| `access_token` | `str` | Personal access token or OAuth token. | None |
| `auth_type` | `str` | `"auto"`, `"pat"`, or `"bearer"`. `"auto"` selects based on token format. | `"auto"` |
| `depth` | `int` | Traversal depth limit for node tree parsing. | `None` (full tree) |
| `max_requests` | `int` | Maximum allowed API calls in one sync run before cleanly checkpointing. | `1500` |
| `service` | `Any` | Pre-built client for testing or custom transport injection. | `None` |

## What is Ingested

Each file sync yields:

1. **File Overview Document (`id="file:<file_key>"`)**:
   - Title: File name.
   - Content: File metadata, version, last modified date, components catalog (names and descriptions), and file-level comments.
   - Node Set: `figma:<file_key>`.
   - Structure: Top-level document.

2. **Page Documents (`id="page:<file_key>:<canvas_id>"`)**:
   - Title: `<File Name> / <Page Name>`.
   - Content: Frame and Section hierarchies, text node strings (`characters`), component references, and node-targeted comments.
   - Node Set: `figma:<file_key>`.
   - Structure: Child document linked to `file:<file_key>` via Cognee's `STRUCTURE_COLUMN` (`child_of` edge).

## Incremental Synchronization & Deletions

- **Modification Detection**: On each run, the file's `lastModified` timestamp is checked against the saved state in `dlt.current.resource_state()`. If unchanged, document re-extraction is skipped.
- **Tombstones / Deletion**: When a page or component is removed from a file, or when a file is removed from `file_keys`, a tombstone row (`_deleted=True`) is emitted so Cognee's orphan cleanup prunes it from the graph and vector stores.

## Rate Limits & Error Handling

- On HTTP 429 (`Too Many Requests`), the connector records `failed_rate_limit`, checkpoints progress, and exits without blocking the process or spinning in sleep. The subsequent run resumes.
- Transient 5xx server errors are retried up to 3 times with exponential backoff.
