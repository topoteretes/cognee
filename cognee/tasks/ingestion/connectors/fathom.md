# Fathom Meeting Notes SDK Connector

A native Fathom data-source connector for [cognee](https://github.com/topoteretes/cognee): ingest meeting notes, action items, executive summaries, and speaker transcripts into persistent AI memory.

It exposes a standard `dlt` source passed directly into `cognee.add()` and `cognee.cognify()`. Meeting overviews, key takeaways, action item assignees, and speaker dialogue are extracted into structured markdown documents, synchronized incrementally, and reconciled on deletion.

---

## Quickstart

```python
import cognee
from cognee.tasks.ingestion.connectors import fathom_source

# Ingest Fathom meetings with action items and summaries
await cognee.add(
    fathom_source(
        api_key="your_fathom_api_key",
        include_transcript=False,  # Set to True for full speaker dialogue
    )
)

# Build knowledge graph entities and vector index
await cognee.cognify()

# Ask questions across historical meetings
result = await cognee.recall(
    "What did we decide about the database migration in yesterday's sync?"
)
print(result)
```

---

## Authentication & Setup

1. **Fathom API Key**:
   - In Fathom, navigate to your **Account Settings** → **API**.
   - Generate a new API Key.
   - Fathom uses HTTP header authentication: `X-Api-Key: <your_key>`.

Alternatively, configure credentials via `.env`:

```bash
FATHOM_API_KEY=your_fathom_api_key
FATHOM_BASE_URL=https://api.fathom.video/external/v1
FATHOM_INCLUDE_TRANSCRIPT=false
```

---

## What is Ingested

| Field | Description |
|---|---|
| **Document ID** | `fathom:meeting:<meeting_id>` |
| **Title** | `Meeting: <meeting_title>` |
| **Content** | Structured markdown with meeting metadata, attendees, executive summary, action items with assignees, and optional transcript |
| **URL** | Direct link to the recording in Fathom (`https://fathom.video/share/...`) |
| **Node Scope** | `cognee_node_set = ["fathom:meetings"]` |

### Meeting Intelligence Extracted
- **Executive Summary**: Overview paragraph and bulleted key takeaways extracted by Fathom AI.
- **Action Items & Owners**: Action items with checkboxes (`[x]` for completed, `[ ]` for pending) and assignee mentions (`@name`).
- **Speaker Transcripts**: Opt-in speaker-attributed transcript with timestamps (`MM:SS`).

---

## Incremental Synchronization & Deletions

1. **`created_after` Cursor & Rolling Recheck Window**:
   Fathom's API provides `created_after` filtering. Because action items can be checked off days after a call, the connector uses a rolling `recheck_days` window (default 7 days) to ensure recent action-item completions are refreshed in memory without downloading your entire historical archive.
2. **Deletion Reconciliation**:
   When previously seen meeting recordings are deleted upstream in Fathom, the connector compares the active listing against stored state and emits `_deleted=True` tombstones, triggering Cognee's `orphan_cleanup` to remove obsolete calls from the memory graph.
