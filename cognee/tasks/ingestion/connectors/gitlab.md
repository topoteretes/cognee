# GitLab SDK connector

A GitLab data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn a project's issues, merge requests and discussion threads into persistent AI agent memory.

It exposes a `dlt` source you hand straight to `cognee.remember(...)`, reusing
cognee's DLT document ingestion pipeline. One source maps to one GitLab project.
The sync is incremental, resumable, rate-limit resilient, and supports both
GitLab.com and self-hosted GitLab CE/EE instances.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import gitlab_source

await cognee.remember(
    gitlab_source(
        project_id="my-group/my-project",
        access_token="<personal_or_project_access_token>",
        # Optional: For self-hosted instances:
        # base_url="https://gitlab.example.com",
    ),
    dataset_name="gitlab_my_project",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, incremental upsert
    max_rows_per_table=0,
)
```

Use either the project's numeric ID (e.g., `123456`) or its URL-safe path (`"my-group/my-project"`).
Re-running against the same dataset syncs only what changed since the last run.

> **`write_disposition="merge"` is mandatory.** With the default `"replace"`, subsequent
> syncs would drop and recreate the underlying graph documents.

## What is ingested

| Row | Id | Title | Content |
| --- | --- | --- | --- |
| Issue | `issue:<project_id>:<iid>` | `[Issue #12] Title` | State, author, assignees, milestone, labels, URL, description, and folded user discussion notes |
| Merge Request | `mr:<project_id>:<iid>` | `[MR !45] Title` | State, draft status, source/target branches, author, reviewers, labels, URL, description, and review comments |

Rows are flat `{id, title, content, url, _deleted}`.
Discussion notes are folded into their parent issue or merge request document so an AI agent
asking about code review feedback or bug reports can recall the full conversational context.

- **System notes are filtered out**: Automated status changes (e.g., "mentioned in commit", "assigned to")
  are ignored so the Knowledge Graph focuses purely on actionable human discussion.
- **Closed items are included by default** (`include_closed=True`): Engineering context often relies on
  how past issues were resolved or how prior MRs were merged. Set `include_closed=False` to ingest only
  active work items.
- **Deterministic formatting**: No volatile relative dates ("3 days ago") or local clock timestamps
  are rendered into document text, preventing graph embedding pollution across re-syncs.

## Incremental sync, quota and resuming

GitLab REST API v4 supports ordering by `updated_at` and filtering via `updated_after`.
The connector tracks cursor state (`floor`, `at_floor`, `ceiling`) in `dlt.current.resource_state()`:
- Subsequent syncs request only items updated after the highest timestamp seen previously.
- When an API rate limit (`HTTP 429` or low `RateLimit-Remaining`) or request budget cap is reached,
  the walk stops cleanly, preserves the cursor, and records the event in `cognee_sync_stats` (`failed_rate_limit`).
- DLT commits all fetched items up to the cut-off point, and the next run seamlessly resumes from that cursor.

## Authentication & Permissions

- **Personal Access Tokens (PAT)** / **Project Access Tokens**: Default mode (`private_token=True`).
  Sent via `PRIVATE-TOKEN` header.
- **OAuth Bearer Tokens**: Pass `private_token=False` to send as an `Authorization: Bearer <token>` header.
- **Required Scopes**: `read_api` or `read_repository`.
