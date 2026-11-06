# cognee-connector-slack

Slack connector for [cognee](https://github.com/topoteretes/cognee): the Slack app
cognee ships (`/cognee-ask`, `/cognee-remember`, `/cognee-link`, `/cognee-import`, the
"Remember this" and "Remember entire thread" message shortcuts, the App Home tab), the
channel-history import that turns selected conversations and threads into documents in a
cognee dataset — one document per conversation, idempotent re-imports, native `update` for
edited threads, deletion only for a thread Slack confirms is gone, and an opt-in scheduled
refresh — and the provider's connection contract (`CONNECTOR`) for a cognee server or
Cognee Cloud.

Slack ships no DLT source: the history import is its own flow, driven from Slack's import
dialog or the HTTP API below, and it runs on a cognee server. The pure parts of the
package (`channels`, `home`, `history_models`, `response_url`, `settings`, `connector`,
`webhooks`) import no host; everything that touches cognee's credential store, search or
ingestion lives under `cognee_connector_slack.cognee_host` and is installed with the
`cognee` extra.

## Install

```bash
pip install cognee-connector-slack             # the connector contract and the pure modules
pip install "cognee-connector-slack[cognee]"   # plus the Slack app and history import for a cognee server
```

### Requirements

The `cognee` extra accepts any cognee >= 1.6.5. The history import (`cognee_host.history`,
`history_client`) reports its progress through `cognee.modules.integrations.sync_progress`
when the installed cognee has it (newer than 1.6.6); on an older cognee it runs the same
way with every progress report dropped (`cognee_host/compat.py`), so `register()` and
`routers()` work on both. The tests that read progress snapshots skip on such a host.

## Configuration

Every environment variable the settings class (`cognee_connector_slack.settings`) reads,
with the `SLACK_` prefix. Names never change between cognee and this package.

```bash
# -- Slack ---------------------------------------------------------------------
# Create a Slack app at https://api.slack.com/apps to get these values.
# CLIENT_ID/CLIENT_SECRET: OAuth & Permissions > your app's Basic Information.
# SIGNING_SECRET: Basic Information > App Credentials — verifies inbound
# requests (slash commands, events, interactive callbacks) via X-Slack-Signature.
# REDIRECT_URI: must match a URL registered under OAuth & Permissions >
# Redirect URLs exactly, and point at this server's
# /api/v1/integrations/slack/callback.
# FRONTEND_BASE_URL: where the browser is redirected back to after
# connect/cancel/error (appends ?slack=<outcome> to /integrations).
#SLACK_CLIENT_ID=""
#SLACK_CLIENT_SECRET=""
#SLACK_SIGNING_SECRET=""
#SLACK_REDIRECT_URI="http://localhost:8000/api/v1/integrations/slack/callback"
#SLACK_FRONTEND_BASE_URL="http://localhost:3000"
# HISTORY_SYNC_ENABLED: the in-process worker that refreshes saved history
# selections every six hours. Set to false when an external scheduler calls
# cognee_connector_slack.cognee_host.history_sync.sync_due_slack_history instead.
#SLACK_HISTORY_SYNC_ENABLED="true"
```

Bot scopes the app needs: `commands`, `chat:write`, `im:write`, `channels:read`,
`channels:history`, `groups:read`, `groups:history` (`cognee_connector_slack.connector.BOT_SCOPES`).
Slack never grants new scopes to an existing installation, so a workspace connected with
fewer scopes must reconnect.

## Usage

The connection contract needs no cognee:

```python
from cognee_connectors_core.testing import FakeCredential

from cognee_connector_slack import CONNECTOR

url = CONNECTOR.authorize_url(state)  # the slack.com consent URL
account = await CONNECTOR.exchange_callback(code, {})  # oauth.v2.access, checked for ok: false
installation = CONNECTOR.parse_installation(account)  # team.id, bot token, metadata
scope = await CONNECTOR.sync_scope(credential, tokens, resources=["C0123456789"])
```

Importing history from an embedded SDK application (cognee installed, the Slack
workspace connected through cognee's `/api/v1/integrations/slack` flow):

```python
from cognee_connector_slack.cognee_host.history import import_slack_history
from cognee_connector_slack.history_models import SlackHistoryRequest

result = await import_slack_history(
    team_id,
    SlackHistoryRequest(dataset_id=dataset_id, channel_ids=channel_ids, days=14),
    user=owner,
)
```

Use the HTTP API when cognee already runs in another process: `examples/slack_history.py`
posts the same selection to `POST /api/v1/slack/history/{team_id}/import` (it needs
`httpx`). The full guide, including the Slack app configuration (commands, shortcuts,
request URLs), the import dialog, the API shapes and the sync guarantees, is
[docs/slack-history.md](docs/slack-history.md).

## Running on a cognee server

```python
from cognee_connector_slack.cognee_host import register, routers

register()  # before the app starts; Slack appears in GET /api/v1/integrations/status

for router, prefix, tags in routers():
    app.include_router(router, prefix=prefix, tags=tags)
```

`register()` puts the `SlackIntegration` adapter in cognee's integration registry, so the
generic `/api/v1/integrations/slack/{authorize,callback,disconnect,...}` routes serve the
OAuth flow. `routers()` lists the two routers cognee mounted itself, both under
`/api/v1/slack`: the public, signature-verified webhook receiver (`POST /commands`,
`/events`, `/interactive` — point the Slack app's request URLs at them) and the
authenticated settings routes (`GET/PUT /channels` for the per-channel allowlist,
`POST /link` for `/cognee-link`, and `/history/{team_id}` with `/import` and `/sync`).

Inbound requests are verified with the core `slack_v0` verifier
(`cognee_connector_slack.webhooks`): an HMAC over `v0:{timestamp}:{raw body}` keyed with
`SLACK_SIGNING_SECRET`, checked against `X-Slack-Signature` with a five-minute replay
window on `X-Slack-Request-Timestamp`.

## What is synced

- Channel history and threads the connection owner selects in the import dialog or the
  API: by date window (`days`, or timezone-aware `oldest`/`latest`), by thread links, or
  by structured `threads`. `thread_mode="started"` (default) selects threads started in the
  window; `"active"` also scans older roots with a reply in the window. A selected
  conversation always includes all currently available replies.
- One document per conversation, tagged with the node sets `slack`, `slack:<team_id>` and
  `slack:channel:<channel_id>`; message text, authors, timestamps and source links are
  kept. Cognee's own bot replies, join/leave events and tombstones are skipped.
- Re-imports are idempotent; an edited thread goes through native `update`; only a thread
  Slack confirms is gone (`thread_not_found` / `message_not_found`) is deleted, and only
  when this importer wrote it. A shorter window never deletes older documents.
- Not imported: files and attachments, direct messages and group DMs, channels the bot or
  the connecting user is not a member of. Every page is fetched before any write; API
  errors, missing cursors and the request/message caps (1,000 requests and 50,000
  messages by default, raisable per request up to 10,000 / 500,000) fail the import
  rather than writing a partial snapshot.
- The slash commands and shortcuts store what a member asks for in the member's own
  memory (`/cognee-link` links a Slack member to a cognee account); `/cognee-ask` answers
  privately and posts to the channel only on an explicit Share.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/slack python -m pytest connectors/slack/tests -q
python scripts/check_cognee_free.py cognee_connector_slack
```

`tests/unit` holds the tests moved from cognee plus the contract tests for `CONNECTOR`;
`tests/integration/test_slack_history_native.py` runs the import against real local
stores (SQLite, LadybugDB, LanceDB) with Slack and the LLM mocked. The tests that read
progress snapshots need a cognee that ships `cognee.modules.integrations.sync_progress`
(see Requirements) and skip otherwise.
