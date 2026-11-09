# Bluesky Data-Source Connector

Ingest Bluesky (AT Protocol) posts, author feeds, and social interactions into Cognee memory.

## Capabilities

* **Zero Credentials for Public Data**: Public AT Protocol feeds require no authentication token.
* **Rich Facet Resolution**: Handles links, mentions, and tags within posts, rendering them into markdown links.
* **Thread Context Awareness**: Captures parent reply contexts and quoted posts directly within the document.
* **Incremental Synchronization**: Respects cursor pagination for low-latency chronological polling.
* **Deletion Awareness**: Emits `_deleted: True` tombstones on removed posts when run with `write_disposition="merge"`.

## Quick Start

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import bluesky_source

async def main():
    # Ingest posts from a specific Bluesky handle
    await cognee.remember(
        bluesky_source(
            handle="bsky.app",
            limit=50,
        ),
        dataset_name="bluesky_intelligence",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Search cognee memory
    results = await cognee.search("What are the latest announcements from Bluesky?")
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(main())
```

## Authenticated Usage (Optional)

For private feeds or searching network-wide:

```python
from cognee.tasks.ingestion.connectors import bluesky_source

source = bluesky_source(
    identifier="your-handle.bsky.social",
    app_password="xxxx-xxxx-xxxx-xxxx",
    query="ai agents",
    limit=100,
)
```
