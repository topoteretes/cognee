# Integrations Hub

First-class connectors that pull data from external sources into cognee memory.

Every connector here is built on cognee's **DLT ingestion subsystem**, so they
all share the same guarantees instead of each reinventing ingestion:

- **One call to ingest** — pass the connector's source to `cognee.remember(...)`.
- **Incremental re-sync** — `write_disposition="merge"` upserts by primary key;
  re-running only pulls the delta.
- **Forget-on-source-deletion** — records deleted upstream are removed from the
  graph + vector + relational stores via the shared `orphan_cleanup` path.
- **Optional, lazy dependencies** — install just the extra you need.

## Available connectors

| Source | Extra | Factory | Example | Notes |
|---|---|---|---|---|
| **Gmail** | `cognee[gmail]` | `cognee.tasks.ingestion.connectors.gmail_source` | [`demos/gmail_connector_example.py`](../demos/gmail_connector_example.py) | Messages/threads, label & query scoped. Incremental via `historyId`; trashed/deleted mail forgotten on next sync. OAuth2 (read-only). |
| **Confluence** | `cognee[confluence]` | `cognee.tasks.ingestion.connectors.confluence_source` | [`demos/confluence_ingestion_example.py`](../demos/confluence_ingestion_example.py) | Cloud spaces/pages + footer comments, space-key scoped. Incremental via `version.createdAt`; pages deleted upstream forgotten on next sync. API token (read-only). |

## Gmail quickstart

```bash
pip install "cognee[gmail]"   # or: uv sync --extra gmail
```

```python
import cognee
from cognee.tasks.ingestion.connectors import gmail_source

await cognee.remember(
    gmail_source(label_ids=["INBOX"], credentials_path="credentials.json"),
    dataset_name="gmail_inbox",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,   # 0 = no read cap, so forget-on-delete sees the whole inbox
)

answer = await cognee.search(
    query_text="What did my manager ask me to do this week?",
    datasets=["gmail_inbox"],
)
```

See the [example](../demos/gmail_connector_example.py) for OAuth setup,
incremental re-sync, and the privacy / opt-in notes.

## Confluence quickstart

```bash
pip install "cognee[confluence]"   # or: uv sync --extra confluence
```

```python
import cognee
from cognee.tasks.ingestion.connectors import confluence_source

await cognee.remember(
    confluence_source(
        base_url="https://your-domain.atlassian.net",
        email="you@example.com",
        api_token="…",          # https://id.atlassian.com/manage-profile/security/api-tokens
        space_keys=["ENG"],     # omit to sync every readable space
    ),
    dataset_name="confluence_wiki",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,   # 0 = no read cap, so forget-on-delete sees the whole space
)

answer = await cognee.search(
    query_text="What is our incident response process?",
    datasets=["confluence_wiki"],
)
```

See the [example](../demos/confluence_ingestion_example.py) for token setup and
incremental re-sync.

## Adding a new connector

1. Create `cognee/tasks/ingestion/connectors/<source>.py` exposing a factory
   that returns a `dlt` resource/source (`primary_key`, `write_disposition="merge"`,
   and a `hard_delete` marker column for deletions). Keep the SDK import lazy.
2. Add a matching optional-dependency extra in `pyproject.toml`.
3. Add a runnable example under `examples/demos/` and a row to the table above.
4. Add mocked-SaaS + mocked-LLM tests (no live credentials in CI).
