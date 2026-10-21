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
(pages unchanged since the last sync, neither fetched nor published again), `deleted`, `failed`, `failed_rate_limit`, `failed_budget`,
`unresolved`, `requests` and `complete` (1 only when the whole notebook was read and published).
An empty dictionary means dlt resumed a pending load without extracting this resource: build a
fresh source and run again.

## What is read

Notebook, section and section-group metadata and each page's HTML, through
`GET /me/onenote/...` with `$orderby=createdDateTime asc` and `$top=100` paging. A page becomes
one row:

| Field | Value |
|---|---|
| id | the Graph page id (the dlt primary key) |
| title | the page title, or `Untitled` when it is empty |
| url | the page's `oneNoteWebUrl` link |
| content | the page text |
| cognee_structure | `{"ancestors": [...]}`, nearest parent first: the section, its section groups from the innermost out, then the notebook, each as `{"kind", "id", "name"}` with the Graph id |

The page text is the HTML body as plain text. Images become `Image: <alt text> (<reference>)` and
attachments `Attachment: <name> (<reference>)`; links keep their target in parentheses. **Images
and attachments are not downloaded**, and handwriting or text inside images is not read.

Where a page sits is not part of its text. It travels as `cognee_structure`, which cognee does not
hash, so renaming a notebook, section or section group, or moving a page to another section or
section group, keeps every page's document id, extracts nothing again and only updates the
`child_of` edges (`Notebook <- Section <- Page`; the containers have no row of their own). The
page's `url` is a column, not text.

## Incremental model

Graph has no delta feed for OneNote, so every run lists the whole notebook (metadata only) and
compares each page with what the last complete run published. A page is fetched and published
only when it is new or its fingerprint changed: its `lastModifiedDateTime`, title, link, the
notebook, section groups and section above it, or the renderer version. Unchanged pages are
neither fetched nor yielded (they stay in the staging table, and cognee reads the whole table
back), so a quiet sync stages no rows and extracts nothing. A timestamp that changes while the
text stays the same costs one HTML request and no re-extraction. Renaming or moving a container
changes the fingerprint of every page below it: each costs one HTML request (the text is not
kept), is published again so its structure is written, and is not extracted again.

A run is **all or nothing for the notebook**: documents and deletions are published only after
the complete snapshot was read and validated. Any expected failure (Graph error, rate limit,
spent budget, malformed metadata or an unreadable page) returns normally with `failed > 0` and
`complete == 0`, publishes nothing and keeps what it already learned: the bodies fetched but not
yet published and confirmed removals are checkpointed, so the next run resumes without repeating
them. Bodies are dropped from state the moment a complete run publishes them.

State keys: `connection` (a hash of account and notebook, so another notebook cannot reuse the
name), `onenote_cache.published` (page id to a 24 character fingerprint, no page text),
`onenote_cache.pending` (page id to the fetched text, only between a cut run and the run that
publishes it), `onenote_cache.unresolved`, `onenote_cache.confirmed_missing` and
`onenote_cache.swallowed`. State does not grow with the size of the pages.

If the staging table is deleted while the dlt state is kept, unchanged pages are not published
again and stay missing until they change. `cognee.forget` removes neither, so it does not cause
this.

## Deletion and its gaps

A page that was synced and is missing from a complete listing is not forgotten on that evidence
alone. Its metadata is read once more, and it is forgotten only when

- Graph answers 404 or 410 with code **20113** (Microsoft: "the resource specified in the request
  was deleted"), or
- the page is still there but its parent notebook is another notebook (a positive move).

Confirmations are checkpointed so a large deletion can finish across runs cut by the request
budget, but they are dropped the moment the page shows up again. A carried confirmation is also
dropped, and the page asked about afresh, when a run reads a container as gone (a section,
section group or notebook whose listing answers 20113) that the run which recorded the
confirmation did not. When it is the same container as before, the container itself is asked
about once per run (one extra request each, counted in `max_requests`); if it resolves now, the
carried confirmations are dropped as well. A page restored in between could otherwise be
forgotten because one listing answered wrongly. A notebook that Graph answers 20113 for clears
the pages that were synced from it, once each page is confirmed as above.

**Code 20102 is not proof of deletion, and it does not block the notebook.** Microsoft describes
it as "you requested a resource that doesn't exist", which also fits a page this token can no
longer see (for example after a scope change). A page that answers 20102 is therefore never
forgotten: it keeps its row and its document, and it is counted in the `unresolved` stat. The
rest of the notebook syncs and publishes normally. Such pages are checkpointed in the state and
asked about again only with the request budget that the walk and the deletion checks left over,
oldest check first, so they can never use up the whole budget or hold back the notebook; running
out of budget on them is not a failure. Every run counts all of them in `unresolved`, checked or
not. A page stops being unresolved the moment it is listed again or Graph answers 20113 for it.
Finding out that a page is unresolved costs one request each, so a scope change that hides many
pages at once takes a few runs under `max_requests` (each run records what it learned), the same
as a large deletion.

How the stats read for such a run: `complete` stays 1 (the whole notebook was walked and
published), `failed` stays 0 (nothing went wrong), `deleted` does not count the page, and
`unresolved` holds the number of pages kept this way. A host that must notice them should watch
`unresolved`, because nothing else in the stats changes. The log gets one warning per run with
the count and no ids.

**Not verified:** which of the two codes Graph returns for a page deleted inside a notebook that
still exists was not checked against a live account. If it answers 20113, such a page is
forgotten as above. If it answers 20102, the page keeps its memory (counted in `unresolved`)
until it answers 20113 or you erase it with `cognee.forget` (see Privacy).

Other gaps:

- **A malformed page blocks its notebook.** A page whose `id` or `lastModifiedDateTime` is
  missing or blank, whose `title` is missing or not a string, or whose links are not objects
  fails the run before anything is published. An empty title is fine (`Untitled`), and so is a
  page without a web link or whose HTML has no `<body>` (its whole text is used). Skipping a
  malformed page is not safe: a new page would stay unsynced without notice, and a page already
  synced would either go stale or be removed on partial evidence. Fix or remove the page and
  the next run recovers.
- **Paging can miss a new page for one run.** Pages are read with `$skip` paging. If another
  page is deleted while the walk is between two batches, a new page can shift past the batch
  boundary, and the run still reports `complete == 1`. The next run picks it up. A page that was
  already synced and is missed this way is not forgotten: its check finds it inside this
  notebook and fails the run with "Page moved during traversal".
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
present. The total sleep of a whole sync is capped at `RETRY_WAIT_BUDGET_SECONDS` (60), because
the walk runs under cognee's process-wide staging lock; once it is spent, the next wait stops the
run cleanly (`failed_rate_limit` for a 429) and the sync resumes next time. Redirects are never followed and every
URL, a paging link included, must be `https://graph.microsoft.com/v1.0/me/onenote/...`.

A 401 stops the run with `failed > 0`. Hosts should refresh the token and run again. Errors carry
a status and a Graph code, never a response body or the token. A token is sent with surrounding
whitespace stripped, and one with whitespace inside is refused before any request.

## Privacy

This connector reads personal notes. Nothing is fetched until you construct a source and call
`remember`.

**Where it goes**

- The page text above becomes cognee documents. During cognify it is sent to your configured LLM
  and embedding providers, like any document you add to cognee. Check that you may send your
  notes there, or use local models. Anyone with read access to the target dataset can read them,
  so keep the notebook in its own dataset.
- dlt state stores the account/notebook hash, page ids and a fingerprint per page, plus the parsed
  text of pages that were fetched but not yet published (a cut run). dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the rendered
  text of every page.
- Logs and `cognee_sync_stats` contain counts and statuses. The token is never logged, stored or
  put in an error.

**What forget removes and what it does not**

- `cognee.forget(data_id=...)` or `forget(dataset=...)` removes the documents, graph and vectors,
  but **not** the staged copy of the text or the dlt state, so the next sync under the same
  dataset and `resource_name` builds everything again. Permanent erasure today means deleting the
  page in OneNote (see the 20102 note above) or dropping the notebook from your selection.
- dlt also keeps completed load packages, with the rendered text of every row, under `~/.dlt` of
  the process, outside cognee's data directories. A page that was deleted in OneNote leaves the
  graph but not that copy. This is a core gap, not specific to OneNote.

## Graph shape

Pages are plain text documents with a position in the notebook: after cognify each page has a
`child_of` edge to its section, a section to its section group or notebook, and so on up to the
notebook (`StructureContainer` nodes with the Graph id and name). Entities with the same name
merge across pages. There is no node set; one (for example `onenote:<notebook>`) is hashed into
every document, so it can only be added before a first release without re-extracting everything.
People named in a note are text, not typed entities.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/ cognee/tests/integration/tasks/connectors/
```

No Microsoft account or token is needed. `cognee/tests/unit/tasks/connectors/fake_onenote.py` is
an in-memory Graph served through `httpx.MockTransport`: notebooks, section groups, paged pages,
HTML and not-found answers (20113 for a deleted page, section or notebook by default, 20102 for a
page marked as hidden). `test_onenote_forget.py` and `test_onenote_ingestion.py` run the real
cognee pipeline with the LLM and embeddings mocked. The connector has not been run against a live
Microsoft account.
