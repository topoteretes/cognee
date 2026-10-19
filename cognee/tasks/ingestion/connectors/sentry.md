# Sentry Data-Source Connector

A native Sentry telemetry and error-tracking connector for [cognee](https://github.com/topoteretes/cognee):
ingest production incidents, unresolved error groups, exception types, and culprit stack traces directly into Cognee's persistent memory.

It exposes a `dlt` document source passed directly to `cognee.remember(...)`. Each Sentry issue becomes a structured, searchable markdown document that flows through chunking and LLM entity/relationship extraction.

---

## Capabilities

- **Structured Telemetry Ingestion**: Extracts issue groups, exception types (e.g. `TypeError`, `DatabaseError`), culprit functions (`src/auth/jwt.py:verify_token`), occurrence frequencies, and stack traces.
- **Deterministic Formatting**: Normalizes stack frames and environment tags in deterministic order to ensure byte-identical document content and prevent duplicate graph embeddings across syncs.
- **Granular Graph Scoping**: Automatically tags every document with `cognee_node_set = ["sentry:<org_slug>/<project_slug>"]` for targeted graph search and isolation.
- **Resilient Keyset Pagination**: Traverses Sentry RFC-5988 `Link` headers and gracefully backs off when `RateLimit-Remaining` drops or on HTTP 429 without losing cursor progress in `dlt`.
- **Self-Hosted & SaaS**: Works out of the box with SaaS Sentry (`https://sentry.io`) and custom self-hosted instances.
- **Credential Hygiene**: Strict token masking in `__repr__` and logs to prevent secret leakage.

---

## Quickstart

### 1. Get a Sentry Auth Token

1. Go to **Sentry** $\rightarrow$ **Settings** $\rightarrow$ **Developer Settings** $\rightarrow$ **Custom User Auth Tokens**.
2. Create a token with `project:read`, `event:read`, and `org:read` permissions.

### 2. Ingestion Example

```python
import os
import asyncio
import cognee
from cognee.tasks.ingestion.connectors.sentry import sentry_source

async def main():
    # Configure Cognee memory
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Ingest Sentry issues into Cognee memory
    source = sentry_source(
        organization_slug="my-company",
        project_slug="api-backend",
        auth_token=os.environ["SENTRY_AUTH_TOKEN"],
        query="is:unresolved",  # optional Sentry query filter
        fetch_events=True,       # fetch latest culprit stack traces
    )

    await cognee.remember(source, dataset_name="prod_telemetry")

    # Query Cognee memory with natural language
    answer = await cognee.recall(
        "What caused the recent authentication errors in api-backend?",
        dataset_name="prod_telemetry",
    )
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `organization_slug` | `str` | *required* | The slug of the Sentry organization. |
| `project_slug` | `str` | *required* | The slug of the target Sentry project. |
| `auth_token` | `str` | `None` | Sentry user authentication token. |
| `base_url` | `str` | `https://sentry.io` | Custom base URL for self-hosted Sentry. |
| `query` | `str` | `None` | Optional Sentry search query (e.g. `is:unresolved`, `level:error`). |
| `fetch_events` | `bool` | `True` | Whether to fetch culprit stack traces from the latest event. |
| `max_issues` | `int` | `None` | Cap the maximum number of issues ingested. |
