# Dropbox SDK connector

`dropbox_source` syncs one Dropbox folder into cognee: one document per supported file, kept
current incrementally, with files deleted or moved out of the folder forgotten. It is a
port of the cognee-community connector (PR #316) with its own small httpx client instead of
the `dropbox` package, which needs Python 3.11 or newer since 12.2.0 while cognee supports
3.10.

## Requirements

No extra: the client is httpx and PDF text comes from pypdf, both core dependencies. The
token needs the scopes `files.metadata.read`, `files.content.read` and `account_info.read`:
every run starts with `users/get_current_account`, refuses a token of another account than
`account_id` (`DropboxAccountMismatchError`, counted in `failed_account`: not an expired token,
so reconnecting the same account does not fix it) and detects a team space (see Limits).

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import dropbox_account_id, dropbox_source

source = dropbox_source(
    folder_id="id:a4ayc_80_OEAAAAAAAAAXw",  # or "" for the whole Dropbox
    account_id=dropbox_account_id(token),    # "dbid:..."
    token=token,                             # or a callable returning a fresh one; or service=
    resource_name="dropbox_notes",           # required, one per folder and account
)
await cognee.remember(
    source,
    dataset_name="my_dropbox",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
# A normal return is not proof of a finished sync: read the counts.
assert source.cognee_sync_stats["failed"] == 0 and source.cognee_sync_stats["complete"] == 1
```

- **One source per folder**, named by the folder's id, never its path: a renamed folder keeps
  its table, cursor and documents. Several folders are several sources, each with its own
  `resource_name`. The integration module (`cognee/modules/integrations/dropbox/`) and the
  Cloud pod build them through `ingestion.sync_scopes`, with the resource name derived from the
  user, the account and the folder id. The integration module refuses a folder inside another
  selected folder when the selection is saved (422 from
  `PUT /api/v1/integrations/dropbox/resources`) and again at sync time: the file would be two
  documents. `dropbox_source` sees one folder and cannot check this; any other host (Cloud
  included) must refuse a nested selection itself.
- `folder_id` is `id:...` or `""`. A path (`/Notes`) is refused at construction.
- Give exactly one of `token` and `service`. `token` is a string or a `TokenSource`: called
  before every request attempt for the token to send, and optionally with `invalidate()`,
  which the client calls on a 401 before it retries that request once (see "Limits").
  `service=` takes a `DropboxClient` (`build_dropbox_service(token)`), for hosts and tests;
  it is never closed by the source.
- The integration module passes a `TokenSource` that caches the token on the sync's worker
  thread and asks the credential store again only near its expiry or after a 401. The token
  path therefore reads the credential once per token lifetime; the host's active check
  (`check_active`) still reads it before every request.
- The Cloud `_PROVIDERS` path builds the client from a plain access token
  (`build_dropbox_service`): no renewal and no retry after a 401 (the run is cut with
  `failed_auth` and Cloud renews the token before the next dispatch), and no
  `DROPBOX_MAX_FILE_SIZE_MB`: Cloud passes its own `max_file_size_mb` if it wants a cap.
- `check_active` is called before every request attempt and every wait (in one-second steps),
  and around every row. What it raises stops the run unchanged.
- `max_file_size_mb` (default `None`, no limit) leaves out larger files, counted in
  `skipped_too_large`. The integration module passes `DROPBOX_MAX_FILE_SIZE_MB` (unset or
  blank: no limit; 0, a negative number, `inf` or `nan` stop the server at startup).

## What is read

- `.txt`, `.md`, `.markdown`, `.csv` as text (UTF-8, or UTF-16 with a BOM; NUL bytes mean
  binary and the file is skipped as unparseable).
- `.pdf` through pypdf.
- Paper docs and other non-downloadable files that offer a `markdown` export, through
  `files/export`.
- Everything else is skipped and counted in `skipped_unsupported_type`.
- From the listing entry: the name, the path (for the structure) and `client_modified` /
  `server_modified` (for the date line).

A row is `id` (the Dropbox file id), `title` (the file name), `content` (`Modified:
YYYY-MM-DD`, a blank line, then the extracted text), `url` (`None`: Dropbox has no link to a file without creating a shared
link, which is a write) and `cognee_structure`: the file's folders, nearest first, up to and
including the selected folder, each `{"kind": "folder", "id", "name"}`. cognify turns those
into `StructureContainer` nodes with `child_of` edges. The path is in the structure only,
which is not hashed: renaming or moving a folder re-stages its files with the same text, so
cognee keeps every document and extracts nothing again (the text is downloaded once more to
carry the new position). Renaming a file itself is different: the title is part of the
document cognee builds and hashes, so a renamed file is extracted once more.

The date line is the day the file's content last changed, taken from the listing entry and
never from the clock: `client_modified` (when its author last changed it), or
`server_modified` for a Paper export and for an entry without `client_modified`; with neither
there is no line. It is a UTC date at day precision. The connector keeps each stored file's
`content_hash`: a file listed again with the same bytes keeps the date it was emitted with,
so a re-save of unchanged bytes on any day is not emitted at all (counted `unchanged`), and a
move or rename of it is emitted with its old date. New bytes on a later day re-extract the
file once, under the new date. A downloadable file listed without a `content_hash` is a
malformed listing (the run raises, the state is untouched); a Paper doc has no
`content_hash` and is re-rendered whenever Dropbox lists it. It lets the graph
date what a file says (a `Timestamp` when the model extracts one) and makes "what changed in
March" findable by keyword (`Modified: 2025-03`).

The rendering has a version (`state["render"]`, separate from the state version `v`). When the
code's render version differs from the stored one, the next run lists the folder whole and
re-emits every file under the new rendering. Bumping it therefore costs one full re-extraction
of every synced folder of every account, on each tenant's LLM key. The first run after upgrading from
state without `render` lists each folder whole once for the same reason.

A Paper doc is dated from `server_modified`. If Dropbox moves that time when the doc is moved
or renamed, its date line changes and it is re-extracted once; to be confirmed in the live run.

A search hit carries the chunk text, date line included. Its reference names the document by its stored file name
(`text_<md5>` on this base), not by the Dropbox file name or path, until references carry
titles (SDK-1500). The file name is the record's `label`.

## Incremental model

The first run resolves the folder (`files/get_metadata`), lists it whole by id
(`files/list_folder`, recursive) and keeps the cursor. Later runs read
`files/list_folder/continue` from the cursor. A reset cursor, a cursor Dropbox no longer
accepts, or a folder whose path changed since the last listing (a rename of the selected
folder) is listed whole again and reconciled like a snapshot, which re-downloads the folder.

Dropbox reports a delete by path only, so the state keeps `index` (file `path_lower` to id).
A move arrives as a delete of the old path plus the same id at the new path; deletes are held
until the whole delta is read, so a move is an update, not a delete. State (`dlt` resource
state, version `v` 1): `render`, `account`, `folder`, `path`, `cursor`, `index`, `folders` (folder path
to id, for the structure), `stored` (ids that reached cognee), `rendered` (per stored file its
`content_hash`, path and date line, as last emitted) and `retry` (ids whose last fetch failed).
A delete names a path only: a path that is a file in `index` is forgotten at once, any other
is swept as a possible folder in one pass over `index` per run of consecutive deletes.
Nothing is written to it until the listing is complete, and dlt keeps the previous state when a
run raises.

A failed download waits by id in `retry` while the cursor moves on, and is fetched again on
every run until it succeeds or Dropbox reports it changed or gone. There is no give-up: each
run it is still pending counts in `failed` and `failed_retry`, so the connection stays degraded
and a host sees it.

## Deletion and its gaps

Tombstones are sent after a complete listing, or on a direct answer about the record:

- a file deleted at Dropbox, or moved out of the folder;
- every file under a deleted subfolder;
- a file replaced by another file or a folder at the same path;
- a stored file that is now skipped (emptied, grown past `max_file_size_mb`, renamed to an
  unsupported type, or one Dropbox now refuses to serve: `restricted_content` or `not_file`,
  counted in `skipped_restricted` and not retried);
- a stored file whose download answers not found: a direct check of the record, so it is
  tombstoned in the same run (`skipped_not_found`), before the listing reports the delete;
- every file of a selected folder this source listed before and Dropbox now reports as not
  found or not a folder (deleted, or access to a shared folder lost). This is decided from
  `files/get_metadata` on the folder alone, without a listing. Forgetting a folder
  whose access was lost is a deliberate privacy choice, accepted by Goran on 2026-11-11: data
  the account can no longer read leaves memory, and comes back if access returns and the
  folder is still selected. A folder never listed that is not found raises
  `DropboxFolderNotFoundError` instead, so a wrong id never empties a table.

What never deletes: any listing error (the state stays as it was), a rate limit, a 401, and a
file that keeps failing: it is retried every run and its stored text is kept until Dropbox
reports it changed or gone. The community connector deleted it after three runs; a failure is
no evidence the file went away.

Deselecting a folder (integration module): after a sync, the tables of folders no longer
selected are emptied (`ingestion.retire_resources`), which removes their documents. Each
folder has its own table, named from its id, with no legacy names, so a selected folder's
failure leaves the other tables alone and blocks retirement only when it may have stored
nothing of its own: a listing error, an account mismatch or a source that raised. Files that
keep failing (`failed_retry`, counted per file on every pass) and a folder cut short by a rate
limit or a 401 after its files were worked on do not block it
(`ingestion.retirement_blocked` with `failed_retry` as a gone key and the cut folders as cut
scopes). The sync still reports degraded whenever anything failed.

Gaps:

- `forget(data_id=...)` and `forget(dataset=...)` leave the dlt staging table and the cursor:
  the next sync with an unchanged Dropbox brings nothing new, and the staged rows are read back
  when the dataset is added again. Erasure has to go through Dropbox, or the host empties the
  table (`ingestion.empty_resource`, what deselecting a folder does).
- A restore may come back with a new file id, which is then a new document.
- A file moved between two selected folders is deleted in one table and added in the other, so
  it is extracted once more.
- dlt keeps completed load packages unless `LOAD__DELETE_COMPLETED_JOBS=true`, which cognee
  sets by default; a failed package stays on disk for the retry.

## Limits

- No file-size cap unless the host sets `max_file_size_mb`. Downloads stream into a spooled
  temporary file (8 MB in memory, the rest on disk), but text and PDF files are then read
  whole to extract their text, so memory follows the file size, a few times over, not the
  extracted text. A host that syncs very large files sets `max_file_size_mb`.
- Retries: a 429 waits its `Retry-After` (a missing one waits 1, 2, 4 s) and a 5xx or a network
  error waits 1, 2, then 4 seconds, three retries per request. All waits of a run share one
  budget of 120 s, because the walk holds cognee's process-wide dlt staging lock, which blocks
  every other dlt ingestion in the process (the community connector's value). A wait that does
  not fit what is left cuts the run: rows already yielded stay, `failed_rate_limit` is set, the
  rest is fetched next run.
- A 401 before anything was stored raises `DropboxAuthError`. After rows it cuts the run with
  `failed_auth`; the next run resumes. With a renewing `TokenSource`, Dropbox's AuthError tag
  decides first: `expired_access_token`, `invalid_access_token`, an unlisted tag or a 401
  without a readable body renew the token once (under the per-account lock, `check_active`
  before the retry) and retry the request once; a second 401 is `failed_auth`.
  `missing_scope` and `route_access_denied` raise `DropboxScopeError` (a `DropboxAuthError`,
  so `failed_auth`) that says to reconnect with the needed scopes, with no renewal or retry.
  `user_suspended`, `invalid_select_user` and `invalid_select_admin` are `failed_auth` with no
  renewal or retry. A refresh Dropbox answers `invalid_grant` revokes the connection and stops
  the run as `failed_auth`, keeping the rows already stored.
- `list_folder` pages at 2000 entries, the size Dropbox documents (approximately).
- The `index` holds one path per file of the folder and is written each run.
- **Team spaces are not supported.** Without the `Dropbox-API-Path-Root` header a team member
  sees only their member folder; team folders outside it are not listed. A run detects a team
  space (`root_info` root namespace differs from the home namespace) and reports it once in
  `team_space_unsupported` and a warning; nothing else changes.

`cognee_sync_stats` (counts only, every key from the start of a run): `scanned`, `unchanged`,
`deleted`, `skipped` and its reasons (`skipped_too_large`, `skipped_unsupported_type`,
`skipped_unparseable`, `skipped_empty_content`, `skipped_not_found`, `skipped_restricted`),
`failed` and its reasons
(`failed_auth`, `failed_account`, `failed_listing` (the folder could not be resolved or
listed: an API error, or a folder never listed that is not found), `failed_rate_limit`,
`failed_retry` once per file still waiting for a
retry), `team_space_unsupported` (0 or 1) and `complete` (1 when the run was not cut).

## Privacy

Errors carry the route and the HTTP status only (`DropboxAuthError`, `DropboxRateLimitedError`,
`DropboxApiError`, `DropboxAccountMismatchError`, `DropboxFolderNotFoundError`), raised `from None`; the client's `repr` hides
the token. Logs carry counts only, never a file name, path or id. The dlt state holds every
file's path (`index`, `folders`) and the staging table the rendered text, both under the
server's dlt directory. Text in a file reaches the LLM at cognify; instructions planted in a
file are out of the connector's reach.

## Testing

Unit tests (`cognee/tests/unit/tasks/connectors/test_dropbox_*.py`) run the real client
against `fake_dropbox.py`, an in-memory Dropbox behind `httpx.MockTransport` that answers with
the JSON shapes of the API reference. Integration tests
(`cognee/tests/integration/tasks/connectors/test_dropbox_*.py`) run `cognee.add` and
`cognee.cognify` with the LLM and embeddings mocked, and replay the user scenario: first sync,
unchanged re-sync, edit, cross-folder move, delete, deselect, folder rename, a 401 after rows.
