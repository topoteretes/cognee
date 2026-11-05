# Google Drive SDK connector

A Google Drive data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync a Drive folder (Docs, Sheets, Slides, Office files including legacy `.doc`/`.xls`/`.ppt`,
PDFs and plain text) into memory — incrementally, with forget-on-delete.

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

Office files need a document loader, the `docling` extra (`pip install "cognee[docling]"`,
Docling 2.116 or newer) or `unstructured`. The legacy binary formats `.doc`, `.xls` and `.ppt`
are first converted to Open XML by LibreOffice, so they also need `soffice` or `libreoffice`
on PATH (on Debian or Ubuntu: `libreoffice-writer libreoffice-calc libreoffice-impress`).
Python extras do not install system packages. Restart the process after installing a Python
extra so the loader registry is rebuilt; LibreOffice is looked up on every sync and needs
no restart.

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

## Upgrading format support

Each folder records a `format_version` alongside its Changes cursor in DLT state. On the
first sync after upgrading from an older connector, it scans the selected folder again to
discover unchanged files that the old connector skipped: Slides and Office files (version 1)
and the legacy `.doc`, `.xls` and `.ppt` (version 2).
The resource name, document IDs and staging table stay the same. No reset or reconnect is
needed. A later expansion of supported formats increments `FORMAT_SUPPORT_VERSION`.

The scan reuses recorded successful `modifiedTime` values, downloading only files without
a matching successful version. SDK 1.6.5 did not record those versions, so its first upgrade
scan downloads existing supported files too; unchanged rows retain their content-based IDs.
The new format version, the cursor and the set of formats that could be read are recorded
only when the scan completes without a transient failure. A rate limit or auth interruption
retains partial progress and does not delete files absent from the incomplete listing. An
extraction or load failure uses DLT's existing rollback/retry mechanism.

A file whose format has no registered loader, or a legacy `.doc`, `.xls` or `.ppt` without
LibreOffice, is skipped without a download and counted as `failed_loader_unavailable`; it
does not hold the cursor or the upgrade marker. The folder records which of the formats above
could be read when it was scanned. When a sync finds that one which could not be read now can,
for instance after `cognee[docling]` or LibreOffice was installed, it scans the folder once
more and reads the files it skipped, reusing successful versions. A format that nothing
installed can read therefore never keeps the folder in full-scan mode. Unsupported formats
and permanently unreadable files do not prevent the upgrade either.

A file whose loader is registered but fails because one of its dependencies cannot be
imported is reported the same way, but installing the dependency does not change what can
be read, so no scan is requested: edit the file in Drive or clear the source's state to read
it.

A file the loader fails on for any other reason, such as a corrupt or password protected
`.doc` or `.docx`, is a transient failure: neither loader tells such a failure from a passing
one by exception type. It holds the cursor and the upgrade marker, so while that file exists
the folder is listed in full on every sync and that file is downloaded again. Healthy files
still load and existing content is retained. Docling converts legacy files with an isolated
temporary LibreOffice profile per conversion, so concurrent workers do not share a profile
lock; with `unstructured` that is not the case. LibreOffice rewrites legacy files as Open XML
before parsing, which can change formatting and formula values: the connector ingests
extracted text, not a formula-preserving copy.

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
`.ods`, `.odp`, `.rtf`, `.html`, `.epub`, and the legacy `.doc`, `.xls`, `.ppt`). They go through
cognee's loader engine, the same loaders `add()` uses. Office files, `.rtf` and `.epub` need the
`docling` (or `unstructured`) extra, and the legacy binary formats also LibreOffice; `.html` is
read as plain text without it. Images, audio and video are not read.

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
| `failed_loader_unavailable` | an Office or document file and no loader installed (install `cognee[docling]`), or a legacy `.doc`, `.xls` or `.ppt` and no LibreOffice on PATH | yes |
| `failed_export_too_large` | a Google Doc or Sheet over Google's export limit (about 10 MB) | yes |
| `failed_content_extraction` | the download failed, or the loader failed (a model download, a bug), for a reason that may pass; the file is tried again and a loader failure never forgets what was ingested | yes |

A run that hits a rate limit or an expired token stops cleanly instead of raising: the rows it
fetched are kept (by their `modifiedTime`), nothing is reconciled as absent from the incomplete
listing, the Changes cursor stays, and the next run goes on from there. A refused token before
the run produced anything raises a `DriveAuthError`, which the sync reports as `failed_auth`, so
the connection can be reconnected.

A transient failure (`failed_content_extraction`: a download or loader failure that may pass) keeps the Changes cursor where it was; the next
run fetches only the failed and the changed files. `failed_export_too_large` and
`failed_unreadable` are reported but do not hold the cursor, because a rerun cannot cure
them. `failed_loader_unavailable` does not hold the cursor either and preserves previously
ingested content; installing a loader for the format schedules the scan described above. Fetching is bounded: at most `max_workers` files are in flight, and together at
most 256 MB of reported file size (a larger file runs alone).

When syncing multiple folders into one dataset, assign each a stable, distinct
`resource_name`. It isolates both the Changes API cursor and cleanup scope.
Previously ingested documents without a table provenance stamp are retained
conservatively; explicitly clear and re-sync the dataset to rebuild legacy data.

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

Real Office parsing and upgrade recovery are tested with the slim Docling extra:

```bash
uv run --extra docling pytest cognee/tests/integration/tasks/connectors/test_google_drive_formats.py -v
```

This test downloads fixture bytes through the connector into its temporary files, parses
them with the actual loaders, and merges the rows into a real DLT database seeded with an
older sync checkpoint. It also verifies rollback on an interrupted listing and a no-change
sync after reopening the pipeline. Google Slides exports are mocked; a live Google API
smoke test is still required when validating a Cloud deployment.

The legacy formats are tested separately and need LibreOffice as well:

```bash
uv run --extra docling pytest cognee/tests/integration/tasks/connectors/test_google_drive_legacy_office.py -v
```

They use genuine binary fixtures (`cognee/tests/test_data/legacy_office/`), a real DLT
database, old checkpoints, an interrupted listing, a missing LibreOffice, a corrupt input and
parallel conversions. The Docling slim CI job installs LibreOffice and runs them; elsewhere
the file is skipped when Docling or LibreOffice is missing.
