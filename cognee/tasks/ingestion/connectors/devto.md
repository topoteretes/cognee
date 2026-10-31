# Dev.to SDK connector

A Dev.to data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn an author's or organization's published articles into memory.

It exposes a `dlt` source you hand straight to `cognee.remember(...)`, so it reuses
cognee's DLT ingestion path. One source is one Dev.to (Forem) username. It reads public
data only, so it needs no API key and no extra install.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import devto_source

await cognee.remember(
    devto_source(username="ben"),
    dataset_name="devto",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
)

answer = await cognee.recall("What has Ben written about AI and design?")
```

Run it again with the same dataset to sync only what changed.

> **`write_disposition="merge"` is mandatory.** With the default `"replace"` the second
> sync would drop and reload the table.

## What is ingested

| Row | Id | Title | Content |
| --- | --- | --- | --- |
| Article | `devto:article:<id>` | article title | author, publish date, tags, canonical URL when it points elsewhere, then the article's Markdown |

Rows are flat `{id, title, content, url, _deleted}`. Each tag becomes a node set
`devto:tag:<tag>`, so a search can be narrowed to one topic. Reaction and comment counts
change all the time and are not rendered, so they never cause re-extraction.

- Only **published** articles are read. Drafts need the author's API key and are not
  supported yet.
- Comments are not read.

## Incremental sync and deletion

Every run lists all of the author's articles (1,000 per request). The body of an article
is fetched only when it is new or its `published_at` / `edited_at` stamps changed, so a
quiet run costs one request per thousand articles and yields nothing.

An article that is no longer listed (unpublished or deleted) is forgotten: it gets a
`_deleted` tombstone and its document is removed. One that disappears between the listing
and its fetch is counted in `vanished` and handled the same way.

Dev.to answers an unknown username with an empty list, not an error. So an empty listing
after earlier runs synced articles raises `DevToEmptyListingError` and forgets nothing: a
typo or a renamed account would otherwise wipe the dataset. To really start over, pass a
new `resource_name`.

A listing entry without an id, an article body that does not match its id, a 404 on the
listing and other API errors raise (`DevToAPIError`). The client retries connection errors
and HTTP 5xx three times, and HTTP 429 with the server's `Retry-After` (up to 30 seconds).
Error messages carry status codes and article ids, never response bodies.

After a run, `cognee_sync_stats` holds `synced`, `unchanged`, `deleted` and `vanished`.

## Known gaps

- An edit that Dev.to does not stamp in `edited_at` is not picked up.
- If the author publishes while a multi-page listing is being read, a page boundary can
  shift and one article may be missed for that run; it is tombstoned and comes back on the
  next run.
- Renaming the account changes the default `resource_name`, so the articles are ingested
  again under the new table; the old one stays until it is forgotten by hand.

## Privacy

This connector reads only what Dev.to publishes to everyone. It is **opt-in**: nothing is
fetched until you construct a source and call `remember`. Anyone with read access to the
target dataset can see what was ingested.
