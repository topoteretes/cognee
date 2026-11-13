# Bluesky Connector

Turn public Bluesky feeds, actor timelines, and discussions into searchable memory with Cognee.

The Bluesky connector queries the decentralized [AT Protocol](https://atproto.com/) public AppView (`public.api.bsky.app`), requiring **zero credentials or API keys** for public actors.

---

## Quickstart

Ingest any public Bluesky user timeline directly into Cognee:

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import bluesky_source

async def main():
    # 1. Stream posts from the official Bluesky team account
    source = bluesky_source(actor="bsky.app")

    # 2. Ingest into Cognee memory with merge deduplication
    await cognee.remember(
        source,
        dataset_name="bluesky_memory",
        primary_key="id",
        write_disposition="merge",
    )

    # 3. Recall insights across conversations
    results = await cognee.recall(
        query="What new features did Bluesky release recently?",
        dataset_name="bluesky_memory",
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Features

- **Zero-Auth Public Reads**: Works immediately out of the box against `https://public.api.bsky.app`.
- **Rich Text Facet Resolution**: Accurately resolves links, `@mentions`, and `#hashtags` into clean Markdown using UTF-8 byte offset slicing (preventing multi-byte emoji drift).
- **Thread & Quote Context**: Flattens parent replies and quoted post embeds into the document for complete semantic context during vector search and graph extraction.
- **Incremental Synchronization**: Automatically tracks timestamp floors and post IDs so repeated syncs only ingest new posts.
- **Resilient Rate Limiting**: Gracefully captures HTTP 429 backpressure, preserving state without crashing or freezing background tasks.
- **Merge Deduplication**: Built with `write_disposition="merge"` and `_deleted` support for deterministic document lifecycle management.

---

## Configuration Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `actor` | `str` | *Required* | Bluesky handle (e.g. `bsky.app`, `alice.bsky.social`) or DID (`did:plc:...`). |
| `resource_name` | `str` | `bluesky_<actor>` | Custom DLT table/resource name. |
| `access_token` | `str` | `None` | Optional Bearer token for private/authenticated timelines. |
| `page_size` | `int` | `50` | Number of posts per page request (max 100). |
| `max_requests` | `int` | `100` | Maximum request budget per run to prevent quota exhaustion. |
