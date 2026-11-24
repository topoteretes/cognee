# cognee-connector-google-drive

Google Drive connector for [cognee](https://github.com/topoteretes/cognee): sync a Drive
folder (Docs, Sheets, Slides, PDFs, Office and plain-text files) into memory, incrementally,
with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's DLT ingestion
path. Files are ingested as **documents** (routed through normal chunking + LLM graph
extraction) via cognee's self-describing content-column mechanism. The first run captures a
Changes API token and lists the folder; later runs read the changes feed and emit only added
or changed files plus `_deleted` tombstones for removed or trashed ones. The package also
carries the provider's connection contract (`CONNECTOR`: the Google OAuth flow, token refresh
and revocation, the folder picker and the sync scope) for a cognee server or Cognee Cloud, and
a `cognee_host/` adapter that plugs it into a cognee server's
`/api/v1/integrations/google_drive/*` routes.

## Install

```bash
pip install cognee-connector-google-drive             # the connector and its DLT source
pip install "cognee-connector-google-drive[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

The source needs a cognee whose ingestion scopes DLT document cleanup by table
(`DOCUMENT_SYNC_VERSION >= 1`, cognee 1.6.5 or later); an older host is refused with a clear
message when the source is built. Office and document files (`.docx`, `.xlsx`, `.pptx`,
`.odt`, `.ods`, `.odp`, `.rtf`, `.html`, `.epub`) are converted to text by cognee's loader
engine, so reading them needs cognee at call time (the `cognee` extra); without it the module
still imports and such a file is reported as `failed_loader_unavailable`.
`cognee-connector-google`, the OAuth client and contract the three Google connectors share,
is installed with the package.

## Configuration

Every environment variable the settings class
(`cognee_connector_google_drive.settings.GoogleDriveSettings`) reads, with the
`GOOGLE_DRIVE_` prefix, plus the ones `google_drive_source` reads for arguments left as
`None`. Names are unchanged from cognee.

```bash
# Google Drive: one OAuth client in a Google Cloud project, consented per
# account. Personal and Workspace accounts use the same client and the same
# flow; they are told apart by the hosted-domain claim after connecting.
# CLIENT_ID / CLIENT_SECRET: from the OAuth 2.0 Client ID (type: Web
# application) in that project's Credentials page.
# STATE_SECRET: any long random string. It signs the OAuth state parameter,
# which is the only thing binding a callback to the user who started it.
# REDIRECT_URI: must be listed verbatim as an authorized redirect URI on the
# client, and point at this server's callback.
# FRONTEND_BASE_URL: where the browser lands after connect/cancel/error
# (appends ?google_drive=<outcome> to /integrations).
# Note: drive.readonly is a restricted scope, so a public launch needs
# Google's OAuth verification first. Test users work without it.
#GOOGLE_DRIVE_CLIENT_ID=""
#GOOGLE_DRIVE_CLIENT_SECRET=""
#GOOGLE_DRIVE_STATE_SECRET=""
#GOOGLE_DRIVE_REDIRECT_URI="http://localhost:8000/api/v1/integrations/google_drive/callback"
#GOOGLE_DRIVE_FRONTEND_BASE_URL="http://localhost:3000"
```

| Variable | Read by | Purpose |
| --- | --- | --- |
| `GOOGLE_DRIVE_CLIENT_ID` | settings | The OAuth client id of the Drive consent screen |
| `GOOGLE_DRIVE_CLIENT_SECRET` | settings | Its client secret |
| `GOOGLE_DRIVE_REDIRECT_URI` | settings | The callback URL, listed verbatim on the client |
| `GOOGLE_DRIVE_FRONTEND_BASE_URL` | settings | Where the browser lands after the OAuth round-trip |
| `GOOGLE_DRIVE_STATE_SECRET` | settings | Signs the OAuth `state` parameter (cognee host) |
| `GOOGLE_DRIVE_FOLDER_ID` | `google_drive_source` | The folder to sync when `folder_id` is not passed |
| `GOOGLE_DRIVE_AUTH_MODE` | `google_drive_source` | `service_account` (default) or `oauth` |
| `GOOGLE_DRIVE_CREDENTIALS_PATH` | `google_drive_source` | The service-account key or OAuth client-secret JSON |
| `GOOGLE_DRIVE_TOKEN_PATH` | `google_drive_source` | Where the cached OAuth user token is read and written (`oauth` mode) |
| `GOOGLE_DRIVE_INCLUDE_SUBFOLDERS` | `google_drive_source` | Recurse into subfolders (default `true`; `false`, `0` or `no` to turn off) |
| `GOOGLE_DRIVE_MAX_FILE_SIZE_MB` | `google_drive_source` | Leave out files larger than this; no limit by default |
| `GOOGLE_DRIVE_MAX_WORKERS` | `google_drive_source` | Threads fetching file content in parallel (default 8) |

The `GOOGLE_DRIVE_*` variables of the source are not settings of the integration: the
settings class ignores them (`extra="ignore"`), and only the SDK source reads them.

## Usage

```python
import cognee
from cognee_connector_google_drive import google_drive_source

await cognee.remember(
    google_drive_source(folder_id="<folder id from the Drive URL>"),
    dataset_name="my_drive_folder",
    primary_key="id",
    write_disposition="merge",  # incremental upsert by file id
    max_rows_per_table=0,  # folders often exceed the default 50-row cap
)

answer = await cognee.search(
    query_text="Summarize the design docs in the shared folder.",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["my_drive_folder"],
)
```

See `examples/google_drive.py` for the full flow. `google_drive_source` also takes
`shared_drive_id` (when `folder_id` is a shared-drive root), `resource_name` (give each folder
its own name when several sync into one dataset, so each folder's Changes API cursor and
cleanup scope stay separate), `check_active` (a host's authorization checkpoint), `service`
(a pre-built Drive client; `build_drive_service_from_access_token` makes one from a token a
host already holds) and `service_factory` (a callable returning a fresh client per fetch
thread, because googleapiclient clients are not thread-safe).

## Setup

1. Enable the Drive API in Google Cloud and create either a **service account** key
   (default, non-interactive) or an **OAuth 2.0 Client ID** (Desktop app). Share the target
   folder with the service account, or authorize the OAuth flow.
2. Point `credentials_path` at the JSON (scope is read-only `drive.readonly`), plus your
   `LLM_API_KEY` like any other cognee run.

`auth_mode="oauth"` uses an interactive browser flow on first run. For headless or CI use,
pre-authorize a token file and point `token_path` at it.

## Running on a cognee server

```python
from cognee_connector_google_drive.cognee_host import register

register()  # before the app starts; the provider appears in GET /api/v1/integrations/status
```

`routers()` in the same module is empty: Drive mounts no routes of its own. The OAuth flow
runs on the generic `/api/v1/integrations/google_drive/...` routes, and the server receives
no webhooks from Drive; a connection syncs on the user's Refresh action.

On a cognee server the integration does three things:

- **Connects a Google account.** The consent asks for `openid email` and `drive.readonly`
  only, with `access_type=offline` and `prompt=consent` so Google issues a refresh token, and
  without `include_granted_scopes`, so this grant is never folded together with a Gmail or
  Calendar one. The credential is keyed on the account's stable Google subject; its email,
  hosted domain and account type (`workspace` or `personal`) are kept as cleartext metadata.
- **Syncs the folders the user picks.** `GET /api/v1/integrations/google_drive/resources`
  offers My Drive, every shared drive and every folder the account can see; `PUT` stores the
  choice under `selected_folder_ids`. A new connection starts with an empty selection and
  syncs nothing; a null selection means everything (My Drive plus every shared drive). Each
  selected folder is one `google_drive_source` with its own DLT resource and Changes cursor
  (a shared drive is listed as a corpus rather than walked as a folder), into the dataset
  `google_drive_<account email>_<hash>`, named as cognee always named it; stale tables are
  retired only after a clean run, and a run that stopped on a refused token is reported as
  `failed_auth` so the connection can be reconnected.
- **Rotates and revokes tokens.** The access token is refreshed five minutes before it
  expires; Google issues no new refresh token, so the stored one is carried forward. An
  `invalid_grant` answer means the account revoked access on Google's side and revokes the
  local credential, since no webhook would ever say so. Disconnecting revokes the grant at
  Google, best-effort.

## How sync + forget-on-delete work

Deletion propagation requires a successful foreground sync (the default).
`cognee.add(..., run_in_background=True)` skips orphan cleanup to preserve existing
memory if the detached ingestion fails. To purge deletions already staged by a
background run, the affected source tables must be loaded again in the foreground;
a no-change incremental retry alone does not guarantee reconciliation.

Incremental sync uses the Drive Changes API page token (persisted in dlt's per-resource state):
the first run captures a start token + does a full folder listing; later runs emit only
added/changed files plus hard-delete tombstones for removed/trashed files. Deletes are emitted
with the `_deleted` hard-delete marker; dlt drops them on `merge` and cognee's `orphan_cleanup`
purges them from the graph, vector, and relational stores. Google Docs/Sheets export to
text/CSV, PDFs are parsed with `pypdf`, and text files download as-is (the formats are listed
below). A PDF that pypdf cannot read (password protected or corrupt) is counted as
`failed_unreadable`; the other ways a file can fail are in the table below.

Every run fetches file content on `max_workers` threads (default 8, env
`GOOGLE_DRIVE_MAX_WORKERS`), each with its own Drive client, and retries rate-limit and server
errors with backoff. The sync state also records each fetched file's `modifiedTime`, so when a
run ends with a failed file (the Changes cursor is then not advanced, unless the failure is one
that a rerun cannot cure, see below) the next run downloads only the failed and the changed
files, not the whole folder again. A host that passes its own `service` gets sequential
fetching; pass `service_factory` (a callable returning a fresh client) to parallelise.

The Drive Changes API is account-wide; deleted-file events carry no metadata and cannot be
scope-filtered, so an incremental run may emit harmless no-op delete rows for files removed
outside the configured folder.

## Which files are not ingested, and how each is reported

There is no file size limit by default. A download is streamed to a temporary file, so memory
depends on the extracted text, not on the file size (the text itself is held in memory by the
staging load, roughly twelve times its size in a measurement with 25 to 300 MB of text).
`max_file_size_mb` (or `GOOGLE_DRIVE_MAX_FILE_SIZE_MB`) is an opt-in limit, checked against the
size Drive reports; a file for which Drive reports no size (Google Docs, Sheets and Slides in
practice) is never left out by it.
A changed limit applies to a file when it is next fetched: on its next change in Drive, or on a
full scan (the first run, or a run after the first one that could not finish its listing). Neither
lowering nor raising it re-reads unedited files by itself; clear the source's state for that.

A file that is in memory from an earlier run and now yields nothing to keep is forgotten, like a
file that left the folder: it grew past your limit, was edited to blank, changed to an
unsupported type, became password protected or corrupt, or became a Google Doc over Google's
export limit. A file that was never
ingested is only counted. A file whose loader is not installed keeps its old content, because
that is a problem with the environment, not with the file.

Formats read: Google Docs (text), Sheets (CSV, first sheet only) and Slides (text), PDF, every
`text/*` file plus JSON and XML, and Office and document files (`.docx`, `.xlsx`, `.pptx`, `.odt`,
`.ods`, `.odp`, `.rtf`, `.html`, `.epub`). They go through cognee's loader engine, the same loaders
`add()` uses, so they need cognee installed (the `cognee` extra). Office files, `.rtf` and
`.epub` need cognee's `docling` (or `unstructured`) extra as well; `.html` is read as plain
text without it. Images, audio and video are not read.

Other files are counted in `cognee_sync_stats`. The log names a file that fails or is over your
size limit; unsupported and empty files are only counted, because a folder of images would log once
per file on every full scan:

| Count | Meaning | Counts as failure |
|---|---|---|
| `skipped_unsupported_type` | Drawings, Forms, images, audio, video, archives, legacy `.doc`/`.xls`/`.ppt` | no |
| `skipped_empty_content` | the extracted text is empty | no |
| `skipped_too_large` | over the `max_file_size_mb` you set | no |
| `failed_rate_limit` | Drive kept answering 429 or a rate-limit 403 after the retries; the run stopped and resumes | yes |
| `failed_auth` | the token was refused (401) after the run had produced rows; the run stopped and resumes | yes |
| `failed_unreadable` | a PDF that downloaded but that pypdf cannot read: password protected, corrupt, or a feature pypdf lacks. The same bytes give the same answer, so it is reported and the cursor still moves. Only memory, the disk and a missing `cryptography` package are treated as passing | yes |
| `failed_loader_unavailable` | an Office or document file and no loader installed (install `cognee-connector-google-drive[cognee]`, and `cognee[docling]` for Office files) | yes |
| `failed_export_too_large` | a Google Doc or Sheet over Google's export limit (about 10 MB) | yes |
| `failed_content_extraction` | the download failed, or the loader failed (a model download, a bug), for a reason that may pass; the file is tried again and a loader failure never forgets what was ingested | yes |

A run that hits a rate limit or an expired token stops cleanly instead of raising: the rows it
fetched are kept (by their `modifiedTime`), nothing is reconciled as absent from the incomplete
listing, the Changes cursor stays, and the next run goes on from there. A refused token before
the run produced anything raises a `DriveAuthError`, which the sync reports as `failed_auth`, so
the connection can be reconnected.

A transient failure (`failed_content_extraction`: a download or loader failure that may pass) keeps the Changes cursor where it was; the next
run fetches only the failed and the changed files. `failed_loader_unavailable`,
`failed_export_too_large` and `failed_unreadable` are reported on the run that meets the file but
do not hold the cursor, because a rerun cannot cure them: after installing the extra, edit the
file or clear the source's state to read it. Fetching is bounded: at most `max_workers` files are in flight, and together at
most 256 MB of reported file size (a larger file runs alone).

When syncing multiple folders into one dataset, assign each a stable, distinct
`resource_name`. It isolates both the Changes API cursor and cleanup scope.
Previously ingested documents without a table provenance stamp are retained
conservatively; explicitly clear and re-sync the dataset to rebuild legacy data.

## The connection contract

`cognee_connector_google_drive.CONNECTOR` is Google Drive on the `cognee-connectors-core`
contract, for a host that drives connectors generically (Cognee Cloud, a future cognee
release): the shared Google descriptor (`cognee_connector_google.google_descriptor`) with
Drive's read-only scope, on `GoogleConnector` for the userinfo lookup that keys the credential
on the Google subject (tier 2), plus the folder picker (`list_resources`: My Drive, the shared
drives, then every folder the account can see) and the sync scope (`sync_scope`). With no
resources named, the scope is the whole account, My Drive plus every shared drive; with
resources, the folders named, the shared drives among them marked in
`options["shared_drive_ids"]` so the source lists each as a corpus. A host that keeps a folder
selection narrows before calling, as the contract has it. The connector refreshes through the
refresh grant, keeps the stored refresh token Google does not rotate, revokes the grant on
disconnect without ever raising, and polls hourly for a host that schedules syncs. Its dataset
is `google_drive_<google subject>`, as Cognee Cloud names it; the cognee server keeps its own
email-based name (see above).

## Privacy

This connector reads the content of the files in the folders you choose. It is **opt-in**:
nothing is fetched until you construct a source and call `remember`, or connect an account to
a server and pick folders. The only scope is `drive.readonly`; the connector never creates,
changes or deletes anything in Drive. Logs and `cognee_sync_stats` carry counts, file names
and HTTP statuses, never file content; keep the service-account key or token file private.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/google:connectors/google-drive python -m pytest connectors/google-drive/tests -q
PYTHONPATH=core:connectors/google:connectors/google-drive python scripts/check_cognee_free.py cognee_connector_google_drive
```

`tests/unit` mocks the Drive API (no live credentials) and covers the sync state machine,
content extraction (with hand-built PDFs in `drive_pdfs.py`), the REST client, the contract
object with Google's token and userinfo endpoints faked, and the cognee adapter and sync;
`tests/integration` runs the source and the example through cognee's real `add()` pipeline
with the LLM and embeddings mocked, which needs cognee installed.
