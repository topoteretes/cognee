# PagerDuty connector

A PagerDuty data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn your team's operational incidents, responder notes, and post-incident timelines into AI memory.

It exposes a `dlt` resource you pass directly to `cognee.remember(...)`, integrating seamlessly into
cognee's DLT ingestion pipeline. The sync is incremental and resumable, and requires no external SDK:
only a PagerDuty API token.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import pagerduty_source

await cognee.remember(
    pagerduty_source(
        api_token="<pagerduty api token>",
        service_ids=["PABC123"],  # Optional: scope to specific PagerDuty service IDs
        include_resolved=True,
    ),
    dataset_name="pagerduty_incidents",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
)
```

Run it again with the same dataset to sync only modified or newly created incidents.

> **`write_disposition="merge"` is mandatory.** With the default `"replace"`, a subsequent
> sync would overwrite previously ingested memory.

## What is ingested

| Entity | ID format | Title | Content |
| --- | --- | --- | --- |
| Incident | `incident:<uuid>` | `INC-124 Title` | Incident status, urgency, priority, service name, escalation policy, assignees, timestamps, description, responder notes, and timeline highlights |

Rows conform to cognee's flat document contract `{id, title, content, url, _deleted}`.
Responder notes and key timeline transitions are embedded directly into their parent incident's document,
allowing AI agents querying about an outage or discussion to retrieve the complete incident context.

- **Resolved incidents**: Ingested by default (`include_resolved=True`), preserving institutional post-mortem memory.
- **Service filtering**: Restrict syncing to specific services (`service_ids=["..."]`).
- **Resiliency**: Built-in exponential backoff on HTTP 429 rate limits, respecting `Retry-After` headers.

## Incremental sync and resumption

PagerDuty sorts incidents chronologically. The connector tracks a state floor (`floor`) and the IDs
observed at that timestamp (`at_floor`). Subsequent runs pass `since=<floor>` with a 5-minute clock-skew
margin to ensure no incident updates are dropped.

Runs are bounded by `max_requests` (default: 1,000 requests per run). If an execution budget is reached,
the run stops gracefully and records its offset in resource state, resuming seamlessly on the next run.

## Authentication and Tokens

The source accepts:
- General Access / Personal REST API tokens (`Token token=...`)
- PagerDuty OAuth user tokens (tokens prefixed with `pd_oauth_` are passed as `Bearer ...`)

Tokens are sanitized and never echoed in error messages, exception strings, or logger payloads.

## Privacy

This connector accesses incident metadata from your PagerDuty workspace. It is strictly **opt-in**:
no data is retrieved until you explicitly instantiate `pagerduty_source(...)` and pass it to `cognee.remember(...)`.
Access to ingested incident memory is controlled by cognee's dataset permissions.
