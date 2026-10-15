# Notion SDK connector

A Notion data-source connector for [cognee](https://github.com/topoteretes/cognee): sync
explicit Notion pages and/or database rows into memory, incrementally, with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Pages are ingested as **documents** (routed through normal chunking + LLM graph
extraction) via cognee's document-row contract, so each one gets a `cognee_node_set` placing it
under its workspace and selected root.

## Requirements

Bundled with the Cognee SDK. Uses `httpx` (core dependency) directly, not `notion-client`, so
no extra install is needed beyond a plain `cognee` install.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import notion_source

await cognee.remember(
    notion_source(
        root_page_ids=["<page id from the page URL>"],
        resource_name="notion_<connection id>",
    ),
    dataset_name="my_notion_space",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)

answer = await cognee.search(
    query_text="What did the roadmap page say about Q3?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["my_notion_space"],
)
```

`resource_name` names the staging table and the incremental state. Every Notion source that
syncs into the same dataset needs its own name: two sources sharing one share state, and each
run then tombstones and forgets the other's pages. A run refuses to continue when the stored
state belongs to a different workspace, but two root sets in one workspace under one name cannot
be told apart from an edited selection. Renaming a source starts it over, and documents synced
under the old name are not cleaned up.

A database (or one of its data sources) works the same way:

```python
notion_source(root_data_source_ids=["<database or data source id>"])
```

## Scope

Only the pages and data sources you name are synced, plus everything nested under them
(sub-pages, and databases embedded in those pages). There is no workspace-wide search; a pod or
UI on top of this connector is expected to pick the roots. Archived and trashed pages are
skipped.

## Incremental model

Notion has no delete feed, so every run walks the selected trees again to find out what is
still there. A page's block content is only re-rendered when its `last_edited_time` or its root
changed since the previous run. Block listing still happens every run, for the page and for
container blocks (columns, toggles, synced blocks, callouts), since that is how nested sub-pages
and databases are discovered. A synced block that copies another page's original is not searched,
since its sub-pages belong to the original: they are only synced when the original's page is
under a selected root. There is no nesting limit: pages, databases and container blocks are
walked and rendered at any depth.

A run costs at least one request per page even when nothing changed: a 20,000-page tree is tens
of thousands of requests, hours at Notion's rate limit of about 3 requests per second. The
connector does not throttle itself; 429s are retried with `Retry-After` or backoff. Pages that
vanished (unshared, trashed, or moved out of the selected roots) are emitted as `_deleted`
hard-delete rows; `dlt` drops them on `merge` and cognee's `orphan_cleanup` forgets them from
the graph, vector, and relational stores.

A transient API error (network failure, or a rate limit/5xx that exhausts the retry budget)
aborts the whole sync before any tombstone is written, so a partial walk can never look like a
mass deletion. A 403/404 when fetching a page or database reached *through* the tree means that
item is gone and is tombstoned; the same error on one of your explicit roots aborts the sync
instead, since that is a configuration problem, not a deletion. A failure while listing a
readable page's blocks or querying a data source also aborts the sync, because it does not prove
anything below it is gone.

Moving a page between two of your selected roots, or changing the roots so that another one
reaches it, changes its `cognee_node_set` (part of the row's identity), so it is re-ingested once
under the new root. That is expected, not a bug.

## node_set

Each row carries `cognee_node_set`: `["notion:<workspace_id>:<root_id>"]`, the workspace and the
root (page, database or data source id) you configured this page under. Always id-based, never
titles. Cognee keeps the name as is, since it already starts with the `notion:` source tag, and
the page's document and chunks belong to that node set. When a page
is reachable from two selected roots, the root nearest to it wins.

The position inside the tree (parent page, database, block) is not recorded yet.

Database row pages also get their properties (title, rich_text, select, multi_select, status,
date, people, number, checkbox, url, email, relation ids) rendered as `Name: value` lines at the top
of the document's content. Relation and people values are the raw Notion ids/names in that
text; they are not resolved into graph edges yet.

## Setup

1. Create a Notion internal integration and copy its token.
2. Share every page/database you want synced with that integration (`... > Connections >
   Add connection`).
3. Set `NOTION_API_KEY`, or pass `token=` directly, plus your `LLM_API_KEY` like any other
   cognee run.

## Limitations

- Teamspaces are not exposed by Notion's public REST API, so pages are grouped by selected
  root, never by teamspace.
- Attachments (file/image/PDF/video blocks) are rendered as a name/caption reference line only;
  the binary is not downloaded in this version.
- A data source with more than 10,000 rows aborts every run: Notion cuts a query off there, and
  syncing only the first 10,000 would forget the rest.
- The walk runs inside cognee's process-wide dlt staging lock, so while a large tree syncs every
  other dlt ingestion in the same process (Drive, Gmail, Linear, CSV) waits.
- A page is only re-rendered when it changes itself. Text it shows from other objects (sub-page
  and database titles, synced copies, mentions) stays as it was until the page is edited.
- Pinned to Notion API version `2025-09-03` (the first with data sources as first-class
  objects). Bumping the pin means re-checking block/property shapes against the new version.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_notion_source.py
uv run pytest cognee/tests/integration/tasks/connectors/test_notion_ingestion.py
```

Both mock the Notion API over `httpx.MockTransport` (no live credentials, no network). The
integration test runs the real source through `cognee.add()`, dlt staging on sqlite and orphan
cleanup; it needs no LLM because it does not cognify.
