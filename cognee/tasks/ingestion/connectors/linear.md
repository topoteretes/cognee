# Linear SDK connector

A Linear data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn a team's issues, comments and projects into memory.

It exposes a `dlt` source you hand straight to `cognee.remember(...)`, so it reuses
cognee's DLT ingestion path. One source is one Linear team. The sync is incremental
and resumable, and needs no extra install: only an access token.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import linear_source

await cognee.remember(
    linear_source(team_id="<team uuid>", access_token="<oauth token or lin_api_ key>"),
    dataset_name="linear",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
)
```

Use the team's id (the uuid), not its key. Run it again with the same dataset to sync
only what changed.

> **`write_disposition="merge"` is mandatory.** With the default `"replace"` the second
> sync would drop and reload the table.

## What is ingested

| Row | Id | Title | Content |
| --- | --- | --- | --- |
| Issue | `issue:<uuid>` | `ENG-12 Title` | state, assignee, priority, labels, project, parent, description, and the issue's comments |
| Project | `project:<uuid>` | project name | status, lead, dates, description, content |

Rows are flat `{id, title, content, url, _deleted}`. A comment is part of its issue's
document, so asking about a discussion finds the issue it belongs to.

- Archived issues, projects and comments are included (`include_archived=True`). Linear
  archives closed issues automatically, so skipping them would drop most of a team's
  history. Issues of a team's sub-teams are not: select the sub-team itself.
- Trashed issues are skipped. Once ingested they stay in memory: this source does not
  forget on delete yet (the `_deleted` column is declared and always `False`).
- A comment that is deleted in Linear is hard-deleted there, so it is not seen and
  stays in the issue's document until the issue changes next.
- An issue moved to another team stays in the old team's table.
- A project shared by several teams is ingested once per selected team.

## Incremental sync, quota and resuming

Linear sorts by `updatedAt`, newest first, and has no change feed. Each of the three
streams (issues, comments, projects) keeps a floor and the ids seen exactly at it, so a
run with no changes yields nothing. Comments are their own stream because an edited
comment does not reliably bump its issue's `updatedAt`: a changed comment re-renders its
issue.

Linear allows 5,000 requests and 2,000,000 complexity points per hour for an OAuth app
user (2,500 and 3,000,000 for a personal API key), shared by everything that user does,
and 10,000 points per query. Pages are capped at 50 (`page_size`) so nested comments stay
under that.

A run is bounded: `max_requests` (default 1,500). It also stops when Linear reports a
rate limit, when the remaining quota falls below a reserve, or when the token expires
after progress was made. It then keeps the state it reached and reports it in
`cognee_sync_stats` (`failed`, plus `failed_rate_limit`, `failed_budget` or
`failed_auth`). It does not raise or sleep, because `dlt` rolls the state of a failed run
back and a big team would never finish. The next run continues where this one stopped.

A walk resumes from the page cursor it stopped at, so even a very large group of rows
sharing one timestamp is read over several runs. A gateway HTTP 429 counts as a rate limit,
like Linear's own `RATELIMITED`.

A token that is rejected before any progress, a team that cannot be found, a pagination
response that does not advance and other API errors raise (`LinearAuthError`,
`LinearTeamNotFoundError`, `LinearAPIError`). Their
messages carry codes only, never the token, variables or a response body.

## Tokens

The source takes a plain access token and never refreshes it. A host that owns the
OAuth flow passes a fresh token per run. OAuth tokens are sent as `Bearer`; personal
API keys (`lin_api_...`) are sent as is. Surrounding whitespace is stripped; a token that is
empty or has spaces or non-ASCII characters is refused when the source is built, without
echoing it.

## Privacy

This connector reads the content of your Linear workspace. It is **opt-in**: nothing is
fetched until you construct a source and call `remember`. Consent covers private teams
the token can see, and anyone with read access to the target dataset can see what was
ingested from them.
