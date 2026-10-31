# Google Drive SDK connector

A Google Drive data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync a Drive folder (Docs, Sheets, PDFs, plain text) into memory — incrementally, with
forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Files are ingested as **documents** (routed through normal chunking + LLM graph
extraction) via cognee's self-describing content-column mechanism.

## Requirements

Bundled with the Cognee SDK, including table-scoped document cleanup and deletion
of the final document. No separate connector package is required.

## Install

```bash
uv pip install "cognee[google-drive]"
# or, from this monorepo:
uv sync --extra google-drive
```

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import google_drive_source

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

See `examples/guides/google_drive.py` for the full flow.

### Host-managed access tokens

For a long-running sync with a host-supplied token, pass its expiry and a synchronous
renewal callback to `build_drive_service_from_access_token`. The callback takes the access
token the client used and returns `(access_token, expires_at)`, with a UTC datetime (a host
serving several clients can return the token another one already renewed). `expires_at` is
required on the client too, and a callback without it is a `ValueError`. Google's
transport invokes it during requests once the token has expired, or when Drive answers
401 (a host that returns the same token then ends the run as an auth failure).
The host keeps refresh secrets and checks that the connection is still authorized.
The callback raises `TokenRenewalError` if it cannot supply a token; the host should
convert every renewal failure to it, since any other exception fails the file being read.

```python
from cognee.tasks.ingestion.connectors.google_drive import build_drive_service_from_access_token


def service_factory():
    return build_drive_service_from_access_token(
        access_token,
        expires_at=expires_at,
        refresh_access_token=renew_access_token,
    )


source = google_drive_source(
    folder_id=folder_id,
    resource_name=resource_name,
    service_factory=service_factory,
)
```

Worker clients call the same callback, so the host must serialize renewal and share the
replacement token. If renewal fails, the source reports an auth failure and preserves
its progress for a later sync. A token without a callback cannot renew during ingestion.

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
text/CSV, PDFs are parsed with cognee's core `pypdf`, and text files download as-is (the formats
are listed below). A PDF that pypdf cannot read (password protected or corrupt) is counted as
`failed_unreadable`; the other ways a file can fail are in the table below.

Every run fetches file content on `max_workers` threads (default 8, env
`GOOGLE_DRIVE_MAX_WORKERS`), each with its own Drive client, and retries rate-limit and server
errors with backoff. The sync state also records each fetched file's `modifiedTime`, so when a
run ends with a failed file (the Changes cursor is then not advanced, unless the failure is one
that a rerun cannot cure, see below) the next run downloads
only the failed and the changed files, not the whole folder again. A host that passes its own
`service` gets sequential fetching; pass `service_factory` (a callable returning a fresh
client) to parallelise.

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
`add()` uses. Office files, `.rtf` and `.epub` need the `docling` (or `unstructured`) extra; `.html`
is read as plain text without it. Images, audio and video are not read.

Other files are counted in `cognee_sync_stats`. The log names a file that fails or is over your
size limit; unsupported and empty files are only counted, because a folder of images would log once
per file on every full scan:

| Count | Meaning | Counts as failure |
|---|---|---|
| `skipped_unsupported_type` | Drawings, Forms, images, audio, video, archives, legacy `.doc`/`.xls`/`.ppt` | no |
| `skipped_empty_content` | the extracted text is empty | no |
| `skipped_too_large` | over the `max_file_size_mb` you set | no |
| `failed_rate_limit` | Drive kept answering 429 or a rate-limit 403 after the retries; the run stopped and resumes | yes |
| `failed_auth` | the token was refused (401) or Google credential renewal failed after the run had produced rows; the run stopped and resumes | yes |
| `failed_unreadable` | a PDF that downloaded but that pypdf cannot read: password protected, corrupt, or a feature pypdf lacks. The same bytes give the same answer, so it is reported and the cursor still moves. Only memory, the disk and a missing `cryptography` package are treated as passing | yes |
| `failed_loader_unavailable` | an Office or document file and no loader installed (install `cognee[docling]`) | yes |
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

The cursor is kept with the folder, shared drive and `include_subfolders` it was taken for. When
any of them changes, the next run lists the whole scope again instead of resuming the old cursor,
so files in a newly included subfolder are read at once. A source synced before this was recorded
has no scope on file and resumes its cursor; it is protected from the first run that ends
normally or stops at Drive's rate limit.

When syncing multiple folders into one dataset, assign each a stable, distinct
`resource_name`. It isolates both the Changes API cursor and cleanup scope. The default
`google_drive_files` is only safe for one folder per dataset.
Previously ingested documents without a table provenance stamp are retained
conservatively; explicitly clear and re-sync the dataset to rebuild legacy data.

## Limits

* Drive's rate limits and server errors are retried with backoff; one that outlasts the retries
  stops the run early (`failed_rate_limit`) and the next run goes on from there.
* A token Drive refuses stops it as `failed_auth`, keeping the rows already read; before any row it
  raises `DriveAuthError`.
* The Changes feed is account wide, so tombstones for files outside the folder are harmless no-ops.
* Fetching is bounded by `max_workers` and 256 MB of reported file size in flight.

## Privacy

File text is sent to the LLM and embedding providers you configured when cognee builds the graph.
`forget()` removes the documents, graph and vectors, but not everything: dlt keeps its staging copy
and the completed load packages, which hold the rendered text of every file, in its own directory
(`~/.dlt` by default, outside cognee's data root), and the next sync under the same dataset and
`resource_name` rebuilds the memory from that state. To erase a file for good, delete it in Drive
(or leave its folder out) and sync, then clear dlt's directory for the source.
The error raised for a failed Drive request carries only its HTTP status, never file names, ids
or tokens. The log lines for a file that was skipped or could not be read do name it (title and
Drive id), so treat the logs as sensitive as the folder itself.

## Setup

1. Enable the Drive API in Google Cloud and create either a **service account** key
   (default, non-interactive) or an **OAuth 2.0 Client ID** (Desktop app). Share the target
   folder with the service account, or authorize the OAuth flow.
2. Point `credentials_path` at the JSON (scope is read-only `drive.readonly`), plus your
   `LLM_API_KEY` like any other cognee run.

## Testing

```bash
uv run --extra google-drive pytest cognee/tests/unit/tasks/connectors/ cognee/tests/integration/tasks/connectors/
```

The unit tests mock the Drive API (no live credentials). They require a cognee build that
includes document-mode (see **Requirements**).
