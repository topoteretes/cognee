# Mattermost SDK Connector

A native Mattermost data-source connector for [cognee](https://github.com/topoteretes/cognee): ingest channels, discussions, and technical decisions into persistent AI memory.

It exposes a standard `dlt` source passed directly into `cognee.add()` and `cognee.cognify()`. Threads are clustered into coherent markdown documents, synced incrementally, and cleaned up automatically when deleted in Mattermost.

---

## Quickstart

```python
import cognee
from cognee.tasks.ingestion.connectors import mattermost_source

# Ingest engineering incident channels into Cognee
await cognee.add(
    mattermost_source(
        mattermost_url="https://mattermost.internal.company.com",
        access_token="pat_your_personal_access_token",
        channels=["incidents", "engineering-architecture"],
    )
)

# Build knowledge graph entities and vector indices
await cognee.cognify()
```

---

## Authentication & Setup

1. **Mattermost URL**:
   The base URL of your self-hosted or cloud Mattermost instance (e.g. `https://chat.mycompany.com`).
2. **Personal Access Token (PAT) / Bot Token**:
   - In Mattermost, go to **Profile** → **Security** → **Personal Access Tokens**.
   - Click **Create New Token**, provide a description (e.g., `Cognee Memory Agent`), and save the generated token.
   - Note: If Personal Access Tokens are not visible, request your workspace admin to enable them under **System Console** → **Integrations** → **Integration Management** (`Enable Personal Access Tokens: true`).

Alternatively, configure credentials via `.env`:

```bash
MATTERMOST_URL=https://mattermost.internal.company.com
MATTERMOST_BOT_TOKEN=pat_your_token_here
MATTERMOST_CHANNELS=incidents,general
```

---

## What is Ingested

| Field | Description |
|---|---|
| **Document ID** | `mattermost:thread:<root_post_id>` |
| **Title** | `Mattermost: #<channel_name> - Thread <root_id>` |
| **Content** | Structured markdown with channel metadata, root message, attachments, and chronological replies |
| **URL** | Direct permalink to the thread in Mattermost (`/_redirect/pl/<root_id>`) |
| **Node Scope** | `cognee_node_set = ["mattermost:<host>/<channel_name>"]` |

### Thread Clustering
A single reply ("Yes, verified and approved") is meaningless without the parent message. This connector automatically clusters replies under their `root_id` into a single, cohesive thread document, preserving full conversational context for downstream agent recall.

### System Events & Audit Noise Filtering
- **System Events**: Joins, leaves, channel header changes, and purpose edits are automatically filtered out.
- **Edit-Audit Copies**: When a post is edited in Mattermost, the server generates an audit clone with `original_id` set. The connector ignores audit clones to prevent phantom duplicate nodes or spurious deletions.

---

## Incremental Synchronization & Deletions

1. **`since` Cursor Synchronization**:
   Mattermost's `/channels/{id}/posts?since=` endpoint is queried using the `update_at` timestamp. Because adding a reply or editing a message bumps the root post's `update_at`, modified threads are automatically fetched and re-rendered.
2. **Safe Millisecond Overlap**:
   To prevent missing messages created in the same millisecond as the cursor boundary, the query uses `since = cursor - 1` and deduplicates previously processed post IDs.
3. **Forget-on-Delete Lifecycle**:
   When posts or threads are deleted upstream in Mattermost, the `since` query returns them with `delete_at > 0`. The connector emits a `_deleted=True` tombstone, triggering Cognee's `orphan_cleanup` to remove the node from the memory graph.
