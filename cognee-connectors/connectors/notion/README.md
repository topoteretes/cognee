# cognee-connector-notion

Notion connector for [cognee](https://github.com/topoteretes/cognee): sync explicit Notion
pages and/or database rows into memory, incrementally, with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's DLT ingestion
path. Pages are ingested as **documents** (routed through normal chunking + LLM graph
extraction) via cognee's document-row contract, so each one gets a `cognee_node_set` placing it
under its workspace and selected root, and a `cognee_structure` saying where it sits in the page
tree. The package also carries the provider's connection contract (`CONNECTOR`: the OAuth flow,
token refresh and revocation, the root picker and the sync scope) for a cognee server or Cognee
Cloud, and a `cognee_host/` adapter that plugs it into a cognee server's
`/api/v1/integrations/notion/*` routes.

## Install

```bash
pip install cognee-connector-notion             # the connector and its DLT source
pip install "cognee-connector-notion[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

The source needs a cognee whose ingestion reads per-row node sets and structure
(`DOCUMENT_SYNC_VERSION >= 3`, cognee 1.6.6rc1 or later); an older host is refused with a clear
message when the source is built. Transport is plain `httpx` for the walk and `aiohttp` for
OAuth, search and parent lookups; `notion-client` is not used.

## Configuration

Every environment variable the settings class (`cognee_connector_notion.settings.NotionSettings`)
reads, with the `NOTION_` prefix. Names are unchanged from cognee.

```bash
# Notion: one public integration in a Notion workspace, authorized per workspace
# (a second workspace is a second connection). Nothing syncs until the user
# picks pages or databases in the integrations page. The browser comes back with
# ?notion=<outcome>.
# CLIENT_ID / CLIENT_SECRET: from the public integration's OAuth settings at
# notion.so/profile/integrations. Notion's token endpoint takes them as HTTP Basic.
# STATE_SECRET: any long random string. It signs the OAuth state parameter.
# REDIRECT_URI: must be listed verbatim on the integration, and point at this
# server's callback.
#NOTION_CLIENT_ID=""
#NOTION_CLIENT_SECRET=""
#NOTION_STATE_SECRET=""
#NOTION_REDIRECT_URI="http://localhost:8000/api/v1/integrations/notion/callback"
#NOTION_FRONTEND_BASE_URL="http://localhost:3000"
```

| Variable | Read by | Purpose |
| --- | --- | --- |
| `NOTION_CLIENT_ID` | settings | The public integration's OAuth client id |
| `NOTION_CLIENT_SECRET` | settings | Its client secret (sent as HTTP Basic to the token endpoint) |
| `NOTION_REDIRECT_URI` | settings | The callback URL, listed verbatim on the integration |
| `NOTION_FRONTEND_BASE_URL` | settings | Where the browser lands after the OAuth round-trip |
| `NOTION_STATE_SECRET` | settings | Signs the OAuth `state` parameter (cognee host) |
| `NOTION_API_KEY` | `notion_source` | An internal integration token, the fallback when `token=` is not passed to the SDK source |

`NOTION_API_KEY` is not a setting of the integration: the settings class ignores it
(`extra="ignore"`), and only the SDK source reads it.

## Usage

```python
import cognee
from cognee_connector_notion import notion_source

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

Setup for the SDK path:

1. Create a Notion internal integration and copy its token.
2. Share every page/database you want synced with that integration (`... > Connections >
   Add connection`).
3. Set `NOTION_API_KEY`, or pass `token=` directly, plus your `LLM_API_KEY` like any other
   cognee run.

## Running on a cognee server

```python
from cognee.api.client import app  # cognee 1.6.x registers its bundled Notion adapter here
from cognee_connector_notion.cognee_host import register

register()  # after that import; the provider appears in GET /api/v1/integrations/status
```

`register()` registers the `NotionIntegration` adapter with cognee's integration registry
(cognee >= 1.6.5), replacing the import-time registration the cognee package had. The order
matters on a cognee that still bundles its own Notion module (1.6.x): importing
`cognee.api.client` imports `cognee.modules.integrations.notion`, whose import registers the
bundled adapter with a plain assignment into the registry, so a `register()` that ran earlier is
replaced by it. Called after that import, as above, this package's adapter is the one the routes
use. Until cognee drops its bundled module the two adapters are the same code; what differs is
the sync behind the Refresh action, which here runs this package's `notion_source` and never
cognee's bundled copy of it (what `ingestion.source_factory("notion")` resolves to on 1.6.x),
so a fix to the source reaches a server as soon as the package is upgraded.

The whole surface is the generic `/api/v1/integrations/notion/*` routes: authorize, callback,
the root picker (`resources`), the Refresh action and disconnect. Notion mounts no routers of
its own and receives no webhooks, so `routers()` returns `[]`. Connecting updates authorization
only; nothing syncs until the user picks roots (`page:<id>` / `data_source:<id>` entries stored
under `selected_roots`), and a sync starts via the Refresh action.

Both the host adapter and `CONNECTOR.dataset_name()` name a connection's dataset
`notion_account_<10 hex of the workspace id>`, as cognee named it before this package existed, so
existing datasets stay where they are and a cognee server and Cognee Cloud land the same workspace
in the same dataset (the formula is `cognee_connector_notion.workspace.dataset_name_for_workspace`).

## What is synced

Only the pages and data sources you name are synced, plus everything nested under them
(sub-pages, and databases embedded in those pages). There is no workspace-wide search; a host or
UI on top of this connector is expected to pick the roots (`CONNECTOR.list_resources` offers the
shared pages and data sources whose parent is not itself shared). Archived and trashed pages are
skipped.

### Incremental model

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

### node_set

Each row carries `cognee_node_set`: `["notion:<workspace_id>:<root_id>"]`, the workspace and the
root (page, database or data source id) you configured this page under. Always id-based, never
titles. Cognee keeps the name as is, since it already starts with the `notion:` source tag, and
the page's document and chunks belong to that node set. When a page is reachable from two
selected roots, the root nearest to it wins.

### Structure

Each row also carries `cognee_structure`: `{"ancestors": [...]}`, where the page sits in the tree,
nearest parent first. A sub-page lists its parent page (`{"kind": "page", "id": ..., "document":
true}`), also when it sits inside a column or toggle: the parent is the page that holds it, not
the block. A database row lists its data source, its database and the page the database sits on;
the database and the data source have no content of their own, so they are listed with their
title (`{"kind": "data_source" | "database", "id": ..., "name": ...}`). A root has an empty list.
A root passed as a database id lists the database; a root passed as a data source id lists the
database it belongs to (read once for its title), but not the page that database sits on.

After each cognify, cognee turns this into `child_of` edges (child to parent, no LLM): a page's
document to its parent page's document, a database row's document to its data source, the data
source to its database, the database to its page. Databases and data sources become
`StructureContainer` nodes, one per dataset. A parent that is not synced (outside the selected
roots, or forgotten) gets no edge. The edges follow the pages: moving a page leaves exactly one
parent edge, deleting one removes its node and edges and leaves its children where they are, and a
database disappears with its last row. The last holds the moment the row is forgotten on a graph
that stores its provenance (one that was empty when cognee first wrote to it); on any other graph
the database goes at the next cognify that still has structured pages. On a graph backend that cannot
delete an edge by identity (Neptune) a moved page keeps its old parent edge as well.

Where a page sits is not part of its identity. Moving it inside its root keeps its document id
and does not run extraction again; renaming a database re-reads its rows once, because the title
is part of what they list.

The first sync after upgrading to a Cognee that reads `cognee_structure`
(`DOCUMENT_SYNC_VERSION >= 3`) re-reads every page once, because the position is new to what is
remembered about it. That costs Notion API requests, not extraction, except for pages whose
rendered text changed in the meantime (see the limitation on text from other objects).

Database row pages also get their properties (title, rich_text, select, multi_select, status,
date, people, number, checkbox, url, email, relation ids) rendered as `Name: value` lines at the top
of the document's content. Relation and people values are the raw Notion ids/names in that
text; they are not resolved into graph edges yet.

### Limitations

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
- Notion sends no webhooks on this path. The contract declares an hourly poll interval for hosts
  that schedule syncs; cognee's own adapter syncs on the user's Refresh action only.

## Development

From the repository root, with the workspace's own interpreter (`uv sync --all-packages --group dev`):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/notion python -m pytest connectors/notion/tests -q
python scripts/check_cognee_free.py cognee_connector_notion
```

`tests/unit` covers the connector, the client, the source's walk and rendering, and the cognee
host adapter and sync (the last two need cognee); `tests/integration` runs the real source through
`cognee.add()`, dlt staging on sqlite and orphan cleanup, and the structure tests through
`cognee.cognify()` over cognee's deterministic mock LLM and embeddings. Both mock the Notion API
over `httpx.MockTransport` (no live credentials, no network).
