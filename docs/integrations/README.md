# Integrations Hub

Data source connectors that plug into cognee's DLT ingestion subsystem
(`cognee/tasks/ingestion/connectors/`). Each connector exposes a
`create_<name>_source(...)` factory returning a dlt resource that you pass
straight to `cognee.remember()`/`cognee.add()` — incremental re-sync
(`write_disposition="merge"`) and forget-on-delete (orphan cleanup) work the
same way across all of them.

| Name | Syncs | Deletion detection | Example | Docs |
|---|---|---|---|---|
| Google Drive | Docs, Sheets, PDFs, text files, and Office docs (docx/xlsx/pptx/odt/ods/odp, requires `cognee[docs]`) in a folder (recursive) | Drive Changes API | [`examples/demos/google_drive_ingestion_example.py`](../../examples/demos/google_drive_ingestion_example.py) | [`cognee/tasks/ingestion/connectors/google_drive/`](../../cognee/tasks/ingestion/connectors/google_drive/) |

## Adding a connector

1. Create `cognee/tasks/ingestion/connectors/<name>/` with a
   `create_<name>_source(...)` factory (see the Google Drive connector for
   the pattern: lazy-imported auth, a dlt resource with incremental state,
   deletions surfaced as merge hard-delete rows).
2. If the source yields unstructured content (documents, messages, pages)
   rather than tabular rows, pass `dlt_content_column=...` to
   `cognee.remember()` so rows get normal chunking + LLM graph extraction
   instead of the relational-row treatment.
3. Add a runnable example under `examples/demos/`, mocked tests under
   `cognee/tests/unit/tasks/` and `cognee/tests/integration/tasks/`, and a
   row to the table above.
