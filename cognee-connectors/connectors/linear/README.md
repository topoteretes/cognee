# cognee-connector-linear

Linear connector for [cognee](https://github.com/topoteretes/cognee): turn a team's issues,
comments and projects into memory, and let a Linear workspace ask that memory back.

It exposes a `dlt` source you hand straight to `cognee.remember(...)`, so it reuses cognee's
DLT ingestion path. One source is one Linear team: its issues, with their comments folded into
the issue's document, and its projects, as flat document rows. The sync is incremental (by
`updatedAt`, with a cursor per stream) and resumable (a run cut short by Linear's hourly quota
continues where it stopped). It does not forget on delete yet: trashed issues are skipped, and
once ingested they stay in memory.

The package also carries the provider's connection contract (`CONNECTOR`, for a cognee server
or Cognee Cloud) and, for a cognee server, the Linear *agent* integration: an app user that
workspace members @mention or delegate issues to, answered from cognee memory.

## Install

```bash
pip install cognee-connector-linear             # the connector and its DLT source
pip install "cognee-connector-linear[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

## Configuration

Every environment variable the settings class (`cognee_connector_linear.settings`) reads, with
the `LINEAR_` prefix. Names never change between cognee and this package. The SDK source needs
none of them: it takes the access token as an argument.

```bash
# -- Linear --------------------------------------------------------------------
# Create a Linear OAuth app at https://linear.app/settings/api/applications.
# This is an *agent* app: the authorize URL uses actor=app, which installs an
# app user into the workspace that members can @mention or delegate issues to
# — the agent answers from cognee memory. On the app, enable "Agent session
# events" (the agent) and "Issues", "Comments" and "Projects" (keep memory
# current), and point its webhook URL at this server's
# /api/v1/integrations/linear/events.
# Connecting ingests every team the app was granted, with full history. Narrow
# it with PUT /api/v1/integrations/linear/resources (GET lists the teams).
# CLIENT_ID/CLIENT_SECRET: from the app's settings page.
# WEBHOOK_SECRET: the app's webhook signing secret — verifies inbound
# deliveries via the Linear-Signature header, and signs the OAuth state
# parameter.
# REDIRECT_URI: must match a callback URL registered on the app exactly, and
# point at this server's /api/v1/integrations/linear/callback.
# FRONTEND_BASE_URL: where the browser is redirected back to after
# connect/cancel/error (appends ?linear=<outcome> to /integrations).
#LINEAR_CLIENT_ID=""
#LINEAR_CLIENT_SECRET=""
#LINEAR_WEBHOOK_SECRET=""
#LINEAR_REDIRECT_URI="http://localhost:8000/api/v1/integrations/linear/callback"
#LINEAR_FRONTEND_BASE_URL="http://localhost:3000"
# A first sync of a big workspace outlasts one hour of Linear's quota, so a run
# stops cleanly and a worker continues it later. RESUME_SYNC_ENABLED turns that
# worker off; RESUME_INTERVAL_SECONDS (at least 60) is how often it looks for a
# connection to continue.
#LINEAR_RESUME_SYNC_ENABLED="true"
#LINEAR_RESUME_INTERVAL_SECONDS="600"
```

## Usage

```python
import cognee
from cognee_connector_linear import linear_source

await cognee.remember(
    linear_source(team_id="<team uuid>", access_token="<oauth token or lin_api_ key>"),
    dataset_name="linear",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
)
```

Use the team's id (the uuid), not its key. Run it again with the same dataset to sync only what
changed.

> **`write_disposition="merge"` is mandatory.** With the default `"replace"` the second sync
> would drop and reload the table.

## Running on a cognee server

```python
from cognee_connector_linear.cognee_host import register

register()  # before the app starts; the provider appears in GET /api/v1/integrations/status
```

`routers()` in the same module is empty: Linear mounts no routes of its own. The OAuth flow
runs on the generic `/api/v1/integrations/linear/...` routes, and every webhook delivery —
agent session events included — arrives on `POST /api/v1/integrations/linear/events`, where
it is verified (`Linear-Signature` HMAC over the raw body, then the body's `webhookTimestamp`
replay window) before it is handled.

On a cognee server the integration does three things:

- **Seeds and keeps memory current.** Connecting a workspace ingests every team the app was
  granted into the dataset `linear_<workspace url key>` (one DLT table per team); `GET`/`PUT
  /api/v1/integrations/linear/resources` narrows that to a selection. Issue, comment and
  project webhooks re-sync the teams they name once the first full pass is done.
- **Answers agent sessions.** An @mention or a delegated issue opens an agent session; the
  handler acknowledges it within Linear's 10-second window, searches the owner's memory
  (`HYBRID_COMPLETION`) and posts the answer as the session's response.
- **Continues cut-short syncs.** A full pass that stopped on the rate limit or the request
  budget is resumed by a periodic worker, `linear_resume_lifespan` in
  `cognee_connector_linear.cognee_host.resume_sync`, an ASGI lifespan for the host to mount.
  cognee 1.6.x's integrations router mounts its own bundled copy of that worker; a host
  without one mounts this package's.

Access tokens last 24 hours and every refresh rotates the refresh token; the adapter refreshes
an expiring token ahead of use, one refresh at a time per connection, and revokes both tokens
at Linear on disconnect.

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

## The connection contract

`cognee_connector_linear.CONNECTOR` is Linear on the `cognee-connectors-core` contract, for a
host that drives connectors generically (Cognee Cloud, a future cognee release): a tier-2
descriptor — the agent-app authorize URL (`actor=app`, scopes
`read,write,app:assignable,app:mentionable`), the token endpoint, and the paths into the
token response — plus one hook that spends the fresh token on a GraphQL query for the viewer
and the organization, because Linear's token response does not say which workspace
authorized and webhooks route by the organization id. It verifies webhooks the same way the
server does, names the same `linear_<url key>` dataset, lists the workspace's teams for a
picker, builds a per-team sync scope, rotates tokens through the refresh grant and revokes
them on disconnect.

## Privacy

This connector reads the content of your Linear workspace. It is **opt-in**: nothing is
fetched until you construct a source and call `remember`, or connect a workspace to a server.
Consent covers private teams the token can see, and anyone with read access to the target
dataset can see what was ingested from them.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/linear python -m pytest connectors/linear/tests -q
python scripts/check_cognee_free.py cognee_connector_linear
```

`tests/unit` covers the source, the client, the contract object and the cognee adapter with
Linear faked; `tests/integration` runs the source and the server's sync through cognee's real
`add()` pipeline, which needs cognee installed.
