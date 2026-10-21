# OneNote SDK connector

A OneNote data-source connector for [cognee](https://github.com/topoteretes/cognee): sync one
selected notebook (sections, nested section groups and pages) into memory through Microsoft
Graph, incrementally, with forget-on-delete where Graph gives proof of a deletion.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Every page becomes a text document that goes through normal chunking and LLM
graph extraction.

## Requirements

Bundled with the Cognee SDK. It uses `httpx` and `beautifulsoup4`, both core dependencies, so no
extra install is needed.

You bring a **delegated** Microsoft Graph access token with `Notes.Read` (OneNote's API does not
support app-only access). The connector never logs in and never refreshes a token: pass a token
string, or a zero-argument callable that returns a valid token for the same account on every
call. Only the public-cloud `graph.microsoft.com` endpoint is supported.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import list_onenote_notebooks, onenote_source

notebooks = list_onenote_notebooks(token)  # [{"id": ..., "displayName": ...}, ...]

source = onenote_source(
    token,
    notebook_id="<notebook id>",
    account_id="<stable account and tenant id, not the token>",
)
await cognee.remember(
    source,
    dataset_name="my_onenote",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
)

# A normal return is not proof of a finished sync: read the counts.
stats = source.cognee_sync_stats
assert stats["failed"] == 0 and stats["complete"] == 1
```

Run the same call again whenever you want to sync. One source is one notebook: loop over notebook
ids to sync several, each with its own source. `list_onenote_notebooks` only reads notebook
metadata, never page content.

| Argument | Meaning | Default |
|---|---|---|
| `token` | Access token string or a zero-argument callable returning one | required |
| `notebook_id` | The notebook to sync | required |
| `account_id` | Stable, non-secret id of the account and tenant (never the token or a display name) | required |
| `resource_name` | Names this connection: it keys the staging table, the sync state and the cleanup of deleted pages. Normalized to dlt's snake-case table name, must not be empty | `onenote_<24 hex of account_id and notebook_id>` |
| `http_client`, `service` | Two names for one pre-built `httpx.Client` (tests, proxies). Give at most one | a client built for the run |
| `check_active` | Host checkpoint, called before every request and before every row is published | none |
| `max_requests` | Budget of HTTP attempts in one run, retries included | 300 |

Every OneNote source that syncs into one dataset needs its own `resource_name` (the default
already differs per account and notebook). A run refuses to continue when the stored state
belongs to another account or notebook. Renaming a source starts it over, and documents synced
under the old name are not cleaned up.

After a run, `source.cognee_sync_stats` holds counts only: `scanned` (pages seen), `skipped`
(page bodies reused from the cache), `deleted`, `failed`, `failed_rate_limit`, `failed_budget`,
`unresolved`, `requests` and `complete` (1 only when the whole notebook was read and published). An empty
dictionary means dlt resumed a pending load without extracting this resource: build a fresh
source and run again.

## What is read

Notebook, section and section-group metadata and each page's HTML, through
`GET /me/onenote/...` with `$orderby=createdDateTime asc` and `$top=100` paging. A page becomes
one row:

| Field | Value |
|---|---|
| id | the Graph page id (the dlt primary key) |
| title | the page title |
| url | the page's `oneNoteWebUrl` link |
| content | `OneNote: <notebook> > <section groups> > <section>`, `Source: <url>`, a blank line, then the page text |

The page text is the HTML body as plain text. Images become `Image: <alt text> (<reference>)` and
attachments `Attachment: <name> (<reference>)`; links keep their target in parentheses. **Images
and attachments are not downloaded**, and handwriting or text inside images is not read.

The breadcrumb is part of the text, so renaming a notebook, section or section group, or moving a
page to another section, re-extracts every affected page once.

## Incremental model

Graph has no delta feed for OneNote, so every run lists the whole notebook (metadata only) and
fetches the HTML of a page only when it is new, its `lastModifiedDateTime` changed, or the
renderer version changed. Unchanged bodies come from a cache kept in dlt resource state (each
page's text once, plus its timestamp). Current metadata is applied on top, so renames and moves
refresh the breadcrumb without an HTML request. A timestamp that changes while the text stays
the same costs one HTML request and no re-extraction.

A run is **all or nothing for the notebook**: documents and deletions are published only after
the complete snapshot was read and validated. Any expected failure (Graph error, rate limit,
spent budget, malformed metadata or an unreadable page) returns normally with `failed > 0` and
`complete == 0`, publishes nothing and keeps what it already parsed: page bodies and confirmed
removals are checkpointed, so the next run resumes without repeating them.

State keys: `connection` (a hash of account and notebook, so another notebook cannot reuse the
name), `onenote_cache.bodies`, `onenote_cache.page_ids` and `onenote_cache.confirmed_missing`.

## Deletion and its gaps

A page that was synced and is missing from a complete listing is not forgotten on that evidence
alone. Its metadata is read once more, and it is forgotten only when

- Graph answers 404 or 410 with code **20113** (Microsoft: "the resource specified in the request
  was deleted"), or
- the page is still there but its parent notebook is another notebook (a positive move).

Confirmations are checkpointed and survive a failed or cut run, but are dropped the moment the
page shows up again. A notebook that Graph answers 20113 for clears the pages that were synced
from it, once each page is confirmed as above.

**Code 20102 is not proof of deletion, and it does not block the notebook.** Microsoft describes
it as "you requested a resource that doesn't exist", which also fits a page this token can no
longer see (for example after a scope change). A page that answers 20102 is therefore never
forgotten: it keeps its row, its cached body and its document, and it is counted in the
`unresolved` stat. The rest of the notebook syncs and publishes normally. The page is asked about
again on every run, and it stops being unresolved the moment it is listed again or Graph answers
20113 for it.

How the stats read for such a run: `complete` stays 1 (the whole notebook was walked and
published), `failed` stays 0 (nothing went wrong), `deleted` does not count the page, and
`unresolved` holds the number of pages kept this way. A host that must notice them should watch
`unresolved`, like the other connectors' `held_back`. The log gets one warning per run with the
count and no ids.

**Not verified:** which of the two codes Graph returns for a page deleted inside a notebook that
still exists was not checked against a live account. If it answers 20113, such a page is
forgotten as above. If it answers 20102, the page keeps its memory (counted in `unresolved`)
until it answers 20113 or you erase it with `cognee.forget` (see Privacy).

Other gaps:

- **A malformed page blocks its notebook.** A page with missing or invalid metadata fails the
  run before anything is published. A page whose HTML has no `<body>` is not malformed: its whole
  text is used. Skipping it is not safe: a new page would stay unsynced
  without notice, and a page already synced would either go stale or be removed on partial
  evidence. Fix or remove the page and the next run recovers from the cache.
- A notebook you leave out of your selection keeps its memory; nothing is forgotten for it.
- Only one active sync per account and dataset is supported.
- A very large notebook can need several runs: the budget counts metadata requests too, so if
  listing alone spends `max_requests`, raise it or reduce the notebook structure.

## Limits

Microsoft Graph throttles OneNote per app and user (delegated): 120 requests per minute and 400
per hour, with five concurrent requests
([service limits](https://learn.microsoft.com/en-us/graph/throttling-limits)). The connector makes
requests one at a time. The default `max_requests=300` stays under the hourly allowance for one
run; it counts every attempt, retries included, and is not an hourly limiter, so several
notebooks or other apps of the same user share the 400.

Graph does not send `Retry-After` for OneNote, so a 429, 500, 502, 503 or 504 is retried up to
five attempts in total with a 1, 2, 4, 8 second fallback; a `Retry-After` header is honoured when
present. The total sleep per request is capped at `RETRY_WAIT_BUDGET_SECONDS` (60), because the
walk runs under cognee's process-wide staging lock; a longer wait stops the run cleanly
(`failed_rate_limit` for a 429) and resumes next time. Redirects are never followed and every
URL, a paging link included, must be `https://graph.microsoft.com/v1.0/me/onenote/...`.

A 401 stops the run with `failed > 0`. Hosts should refresh the token and run again. Errors carry
a status and a Graph code, never a response body or the token.

## Privacy

This connector reads personal notes. Nothing is fetched until you construct a source and call
`remember`.

**Where it goes**

- The page text above becomes cognee documents. During cognify it is sent to your configured LLM
  and embedding providers, like any document you add to cognee. Check that you may send your
  notes there, or use local models. Anyone with read access to the target dataset can read them,
  so keep the notebook in its own dataset.
- dlt state stores the account/notebook hash, page ids, timestamps and each page's parsed text
  (once). dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the rendered
  text of every page.
- Logs and `cognee_sync_stats` contain counts and statuses. The token is never logged, stored or
  put in an error.

**What forget removes and what it does not**

- `cognee.forget(data_id=...)` or `forget(dataset=...)` removes the documents, graph and vectors,
  but **not** the staged copy of the text or the dlt state, so the next sync under the same
  dataset and `resource_name` builds everything again. Permanent erasure today means deleting the
  page in OneNote (see the 20102 note above) or
  dropping the notebook from your selection.
- dlt also keeps completed load packages, with the rendered text of every row, under `~/.dlt` of
  the process, outside cognee's data directories. A page that was deleted in OneNote leaves the
  graph but not that copy. This is a core gap, not specific to OneNote.

## Graph shape

Pages are plain text documents: no node set and no structure. In the Cloud graph view such data
sits in the grey `uncategorized` source; entities with the same name merge across pages. A node
set (for example `onenote:<notebook>`) is hashed into every document, so it can only be added
before a first release without re-extracting everything. People named in a note are text, not
typed entities.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/ cognee/tests/integration/tasks/connectors/
```

No Microsoft account or token is needed. `cognee/tests/unit/tasks/connectors/fake_onenote.py` is
an in-memory Graph served through `httpx.MockTransport`: notebooks, section groups, paged pages,
HTML and not-found answers (20113 for a deleted page, section or notebook by default, 20102 for a page marked as hidden). `test_onenote_forget.py` and `test_onenote_ingestion.py` run the real
cognee pipeline with the LLM and embeddings mocked. The connector has not been run against a live
Microsoft account.
