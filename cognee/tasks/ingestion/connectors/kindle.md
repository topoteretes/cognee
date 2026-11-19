# Kindle SDK connector

A Kindle highlights and notes data-source connector for [cognee](https://github.com/topoteretes/cognee): ingest `My Clippings.txt` into memory as structured books with deduplicated highlights and attached notes.

It exposes a `dlt` source you pass to `cognee.remember(...)`, integrating directly with Cognee's DLT pipeline. Every book becomes one document that is chunked and graph-extracted.

## Features

- **One Document per Book**: Aggregates all highlights and notes belonging to the same book into a clean Markdown document.
- **Overlapping / Extended Highlight Resolution**: When Kindle appends a new clipping for an extended highlight, the connector replaces the older overlapping clipping while preserving any attached notes.
- **Note Attachment**: Notes taken at a location covered by a highlight are attached directly to that highlight; standalone notes are positioned by location order.
- **Bookmarks Skipped**: Kindle bookmarks (which contain no highlight text) are filtered out automatically.
- **Snapshot & Tombstone Support**: If a book's clippings are removed from the file, tombstone records (`_deleted: True`) are emitted to forget the removed book.
- **Safe Failure**: Missing, empty (after previous sync), or malformed clipping entries abort execution to prevent accidental data loss.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import kindle_source

source = kindle_source(clippings_path="~/Documents/My Clippings.txt")
await cognee.remember(
    source,
    dataset_name="kindle_memory",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
    self_improvement=False,
)

assert source.cognee_sync_stats["synced"] > 0
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `clippings_path` | Absolute or `~`-relative path to `My Clippings.txt` | required |
| `resource_name` | Resource identifier for state tracking | `kindle_<sha256_hash>` |
| `check_active` | Optional host authorization checkpoint callback | `None` |
