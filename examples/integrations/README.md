# Data-source connectors

Connectors pull data from external sources (Gmail, Slack, Notion, Google Drive,
Confluence, …) into cognee memory. **Gmail and Google Drive ship in the SDK**;
install their optional extras for Google client libraries and DLT. Other connectors
are distributed under [topoteretes/cognee-community](https://github.com/topoteretes/cognee-community).

Every connector is built on cognee's **DLT ingestion subsystem**, so they all share
the same guarantees instead of each reinventing ingestion:

- **One call to ingest** — hand the connector's `dlt` source to `cognee.remember(...)`.
- **Incremental re-sync** — `write_disposition="merge"` upserts by primary key
  (or `replace` for full-snapshot sources); re-running only pulls the delta.
- **Forget-on-source-deletion** — records removed upstream are deleted from the
  graph + vector + relational stores via the shared `orphan_cleanup` path.
- **Prose ingested as documents** — connectors opt into the document path
  (`dlt_utils.DOCUMENT_SOURCE_ATTR`) so page/message text flows through normal
  cognify (LLM entity extraction), not the relational schema path.

## Available connectors

Install from PyPI; you do **not** need to clone the community monorepo to use them.

| Source | Package |
|---|---|
| Gmail | `cognee[gmail]` |
| Slack (export) | `cognee-community-connector-slack` |
| Confluence | `cognee-community-connector-confluence` |
| Notion | `cognee-community-connector-notion` |
| Google Drive | `cognee[google-drive]` |

## Quickstart (Gmail)

```bash
pip install "cognee[gmail]"
```

```python
import cognee
from cognee.tasks.ingestion.connectors import gmail_source

await cognee.remember(
    gmail_source(label_ids=["INBOX"], credentials_path="credentials.json"),
    dataset_name="gmail_inbox",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,  # 0 = no read cap, so forget-on-delete sees the whole inbox
)

answer = await cognee.search(
    query_text="What did my manager ask me to do this week?",
    datasets=["gmail_inbox"],
)
```

See the SDK [Gmail guide](../../cognee/tasks/ingestion/connectors/gmail.md) and
[Drive guide](../../cognee/tasks/ingestion/connectors/google_drive.md) for setup,
incremental re-sync, and privacy notes. Other connectors are documented in their
community packages.

## Writing a new connector

Publish a `cognee-community-connector-<source>` package (see the ones above as
templates). The connector exposes a factory returning a `dlt` source with a
`primary_key`, a `write_disposition`, and a `hard_delete` marker column for
deletions; for prose sources set `DOCUMENT_SOURCE_ATTR` so rows are ingested as
documents. Keep the third-party SDK a lazy import, and ship mocked-SaaS +
mocked-LLM tests (no live credentials in CI).
