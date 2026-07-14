# Integrations Hub

Data source connectors that plug into cognee's DLT ingestion subsystem
(`cognee/tasks/ingestion/connectors/`). Each connector exposes a
`<name>_source(...)` factory returning a dlt resource that you pass straight to
`cognee.remember()`/`cognee.add()` — incremental re-sync
(`write_disposition="merge"`) and forget-on-delete (orphan cleanup) work the
same way across all of them.

| Name | Syncs | Deletion detection | Example | Connector |
|---|---|---|---|---|
| Google Drive | Docs, Sheets, PDFs, and plain-text files in a folder (recursive) | Drive Changes API | [`examples/demos/google_drive_ingestion_example.py`](../../examples/demos/google_drive_ingestion_example.py) | [`connectors/google_drive.py`](../../cognee/tasks/ingestion/connectors/google_drive.py) |
| Gmail | Messages, label/query-scoped | Gmail History API | [`examples/demos/gmail_connector_example.py`](../../examples/demos/gmail_connector_example.py) | [`connectors/gmail.py`](../../cognee/tasks/ingestion/connectors/gmail.py) |

## Adding a connector

1. Add `cognee/tasks/ingestion/connectors/<name>.py` with a `<name>_source(...)`
   factory (see the Google Drive connector for the pattern: lazy-imported auth,
   a dlt resource with incremental state, deletions surfaced as merge
   hard-delete rows), and export it from `connectors/__init__.py`.
2. If the source yields unstructured content (documents, messages, pages)
   rather than tabular rows, pass `dlt_content_column=...` to
   `cognee.remember()` so rows get normal chunking + LLM graph extraction
   instead of the relational-row treatment.
3. Add a runnable example under `examples/demos/`, mocked tests under
   `cognee/tests/unit/tasks/` and `cognee/tests/integration/tasks/`, and a
   row to the table above.
