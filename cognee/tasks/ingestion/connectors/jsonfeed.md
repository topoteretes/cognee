# JSON Feed Data-Source Connector

Ingest web feeds formatted according to the [JSON Feed specification](https://jsonfeed.org) (v1.0 and v1.1) directly into Cognee's persistent memory and knowledge graph.

## Quickstart

```python
import cognee
from cognee.tasks.ingestion.connectors import jsonfeed_source

# Ingest technical blogs, podcasts, and newsletter feeds
await cognee.remember(
    jsonfeed_source(
        feed_urls=[
            "https://jsonfeed.org/feed.json",
            "https://daringfireball.net/feeds/json",
        ]
    ),
    dataset_name="web_feeds",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)

# Search your indexed feed knowledge
results = await cognee.search("What is the latest update about JSON Feed 1.1?")
print(results)
```

## Features

- **Standard Spec Compliance**: Fully supports JSON Feed v1.0 and v1.1 formats (`items`, `authors`, `tags`, `attachments`, `summary`).
- **HTML & Text Sanitization**: Seamlessly extracts `content_text` or converts `content_html` into clean, normalized markdown text for semantic indexing.
- **Incremental Sync**: Tracks the latest publication timestamp (`date_modified` or `date_published`) per feed, skipping already indexed items on subsequent sync runs.
- **Pagination Support**: Automatically follows `next_url` pagination links up to safety limits.
- **Node-Set Scoping**: Scopes documents to `jsonfeed` and `jsonfeed:<feed_slug>` for targeted retrieval and graph querying.
- **Zero Extra Dependencies**: Relies exclusively on Python's standard library and `dlt`.
