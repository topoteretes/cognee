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
text/CSV, PDFs are parsed with cognee's core `pypdf`, and plain text/markdown/CSV download as-is;
an unparseable file is skipped with a warning.

The first run fetches file content on `max_workers` threads (default 8, env
`GOOGLE_DRIVE_MAX_WORKERS`), each with its own Drive client, and retries rate-limit and server
errors with backoff. The sync state also records each fetched file's `modifiedTime`, so when a
run ends with a failed file (the Changes cursor is then not advanced) the next run downloads
only the failed and the changed files, not the whole folder again. A host that passes its own
`service` gets sequential fetching; pass `service_factory` (a callable returning a fresh
client) to parallelise.

## Which files are not ingested, and how each is reported

There is no file size limit by default. A download is streamed to a temporary file, so memory
depends on the extracted text, not on the file size (the text itself is held in memory by the
staging load, roughly twelve times its size in a measurement with 25 to 300 MB of text).
`max_file_size_mb` (or `GOOGLE_DRIVE_MAX_FILE_SIZE_MB`) is an opt-in limit: a file over it is
counted as `skipped_too_large` and forgotten if an earlier run stored it, like a file that left
the folder. Raising the limit fetches the file again.

Formats read: Google Docs (text), Sheets (CSV, first sheet only) and Slides (text), PDF, every
`text/*` file plus JSON and XML, and Office and document files (`.docx`, `.xlsx`, `.pptx`, `.odt`,
`.ods`, `.odp`, `.rtf`, `.html`, `.epub`). The Office and document files go through cognee's loader
engine, the same loaders `add()` uses, so they need the `docling` (or `unstructured`) extra. Images,
audio and video are not read.

Other files are counted in `cognee_sync_stats`, and the log names each one:

| Count | Meaning | Counts as failure |
|---|---|---|
| `skipped_unsupported_type` | Drawings, Forms, images, audio, video, archives, legacy `.doc`/`.xls`/`.ppt` | no |
| `skipped_empty_content` | the extracted text is empty | no |
| `skipped_too_large` | over the `max_file_size_mb` you set | no |
| `failed_loader_unavailable` | an Office or document file and no loader installed (install `cognee[docling]`) | yes |
| `failed_export_too_large` | a Google Doc or Sheet over Google's export limit (about 10 MB) | yes |
| `failed_content_extraction` | corrupt file, download or parse error | yes |

A failure keeps the Changes cursor where it was; the next run fetches only the failed and the
changed files.

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
