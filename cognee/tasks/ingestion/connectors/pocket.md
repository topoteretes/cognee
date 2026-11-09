# Pocket Reading List Data-Source Connector

Ingest Pocket bookmarks, saved articles, and research tags directly into Cognee's long-term memory layer.

## Capabilities

* **Full Reading List Sync**: Syncs unread articles, archived reading history, and favorites.
* **Tag & Author Preservation**: Retains user tags and publisher authors for structured knowledge graphs.
* **Incremental Synchronization**: Uses Pocket's Unix timestamp `since` filter to pull only newly bookmarked or updated items.
* **Tombstone Deletion Awareness**: Recognizes deleted bookmarks (status `2`) and emits `_deleted: True` under `write_disposition="merge"`.

## Quick Start

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import pocket_source

async def main():
    # Ingest unread reading list articles
    await cognee.remember(
        pocket_source(
            consumer_key="your-pocket-consumer-key",
            access_token="your-pocket-access-token",
            state="unread",
            limit=50,
        ),
        dataset_name="reading_list_memory",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Search through saved reading list
    results = await cognee.search("What articles did I save about LLM agents?")
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(main())
```

## Tag-Filtered Ingestion

```python
from cognee.tasks.ingestion.connectors import pocket_source

source = pocket_source(
    consumer_key="your-consumer-key",
    access_token="your-access-token",
    tag="ai-research",
    limit=100,
)
```
