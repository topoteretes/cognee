# cognee-connector-github

GitHub connector for [cognee](https://github.com/topoteretes/cognee): connects a GitHub org
(or user account) through a **GitHub App installation** and indexes the repositories the
installation covers into cognee's code graph. There is no DLT source and no document
ingestion: a sync is cognee's code route, `remember(content_type="code")`, over the
installation's clone URLs, authenticated with an hour-lived installation token minted on
demand from the app's private key. Nothing durable is stored per installation. Webhooks
keep the graph current: a push to a repository's default branch re-indexes that repository,
repositories added to the installation are indexed, and an uninstall revokes the stored
connection. The package carries the provider's connection contract (`CONNECTOR`, a tier 3
`Connector`) for a cognee server or Cognee Cloud, and the adapter that plugs it into a
cognee 1.6.x server (`cognee_connector_github.cognee_host`).

## Install

```bash
pip install cognee-connector-github             # the connector: app auth, contract, webhooks
pip install "cognee-connector-github[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

## Configuration

Every environment variable the settings class (`cognee_connector_github.settings.GithubSettings`)
reads, with the `GITHUB_` prefix. Names never change between cognee and this package.

```bash
# -- GitHub --------------------------------------------------------------------
# Create a GitHub App at https://github.com/settings/apps to get these values.
# Required app configuration: "Request user authorization (OAuth) during
# installation" enabled; Callback URL pointing at this server's
# /api/v1/integrations/github/callback; Webhook URL pointing at
# /api/v1/integrations/github/events; repository permission
# "Contents: Read-only"; subscribed events: Push, Installation repositories.
# APP_ID: the numeric id from the app settings page.
# APP_SLUG: the app's URL slug (github.com/apps/<slug>).
# APP_PRIVATE_KEY: the app's PEM private key ("\n" escapes accepted).
# WEBHOOK_SECRET: verifies inbound deliveries via X-Hub-Signature-256, and
# signs the OAuth state parameter.
#GITHUB_APP_ID=""
#GITHUB_APP_SLUG=""
#GITHUB_APP_PRIVATE_KEY=""
#GITHUB_CLIENT_ID=""
#GITHUB_CLIENT_SECRET=""
#GITHUB_WEBHOOK_SECRET=""
#GITHUB_FRONTEND_BASE_URL="http://localhost:3000"
```

`GITHUB_REDIRECT_URI` is also read (it is a field of every connector's settings) but the
install flow does not use it: GitHub sends the callback to the URL configured on the app.

| Variable | Used for |
| --- | --- |
| `GITHUB_APP_ID` | `iss` of the app JWT that mints installation tokens |
| `GITHUB_APP_SLUG` | the installation URL the connect flow redirects to |
| `GITHUB_APP_PRIVATE_KEY` | signs the app JWT (RS256; literal `\n` escapes accepted) |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | the user-token exchange that proves who completed the install |
| `GITHUB_WEBHOOK_SECRET` | `X-Hub-Signature-256` verification and the OAuth `state` signature |
| `GITHUB_FRONTEND_BASE_URL` | where the browser lands after the install round-trip |
| `GITHUB_REDIRECT_URI` | unused by GitHub (the callback URL is configured on the app) |

Missing values fail at use time, naming the variable (`GITHUB_APP_SLUG is not configured`).

## Usage

The package ships no DLT source. With the SDK, index repositories directly through
cognee's code route; `app_auth` mints the installation token:

```python
import cognee
from cognee_connector_github import app_auth

installation_id = 12345  # the GitHub App installation, from the connect flow
token, expires_at = await app_auth.mint_installation_token(installation_id)
repositories = await app_auth.list_installation_repository_records(token)

await cognee.remember(
    [app_auth.clone_url(repo["full_name"]) for repo in repositories],
    dataset_name="github_12345",
    content_type="code",
    repo_credentials=token,  # out of band: the clone URLs carry no secret
    raise_on_error=False,
)
print(await cognee.recall("", query_type="CODE", code_query={"operation": "architecture"}))
```

The same steps are what the contract object does for a host:

```python
from cognee_connector_github import CONNECTOR

url = CONNECTOR.authorize_url(state)  # the app-installation URL
record = await CONNECTOR.exchange_callback(code, params)  # verifies installation_id first
installation = CONNECTOR.parse_installation(record)  # provider_account_id = installation id
scope = await CONNECTOR.sync_scope(credential, tokens={})  # clone URLs + an hour-lived token
dispatch = await CONNECTOR.handle_webhook(raw_body, headers)
```

Repositories are identified by their full name (`owner/repo`) in `WebhookDispatch.resources`,
`sync_scope(resources=...)`, `SyncScope.item_resource_ids` and `ConnectorResource.id`.

## Running on a cognee server

```python
from cognee_connector_github.cognee_host import register

register()  # before the app starts; GitHub appears in GET /api/v1/integrations/status
```

`routers()` in the same module is empty: GitHub mounts no routers of its own. The install
flow runs on the generic `/api/v1/integrations/github/...` routes and deliveries arrive on
`POST /api/v1/integrations/github/events`. On that host:

- connecting (and a manual sync) indexes every repository the installation covers into one
  dataset per installation, `github_<account login>` (the same dataset cognee 1.6.x used);
- `push` to the default branch re-indexes that repository, `installation_repositories`
  added indexes the new ones, `installation` deleted or suspended revokes the stored
  connection;
- the sync reports progress through cognee's sync-progress seam when the installed cognee
  has it (newer than 1.6.6); on an older cognee the sync runs the same way without
  progress reports.

## What is synced

- Every repository the installation covers, or the subset a webhook named, through
  cognee's code-graph pipeline: typed code symbols and modules with `calls`/`imports`
  edges, searchable with `SearchType.CODE`. The clone is a shallow copy of the default
  branch.
- Repositories removed from the installation are logged only; their indexed data stays
  until `forget()`.
- Not synced: issues, pull requests, discussions, wikis, other branches.
- Limits: the installation token lives about one hour and repositories are cloned
  sequentially, so a very large installation can outlive the token mid-batch; the affected
  repositories surface as per-repository errors and the next webhook (or a manual re-sync)
  picks them up with a fresh token. A rename on GitHub changes a repository's full name and
  therefore its resource id.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/github python -m pytest connectors/github/tests -q
python scripts/check_cognee_free.py cognee_connector_github
```
