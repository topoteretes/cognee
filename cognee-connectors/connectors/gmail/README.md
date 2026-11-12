# cognee-connector-gmail

Gmail connector for [cognee](https://github.com/topoteretes/cognee): turn your inbox into
memory, "ask my inbox".

It exposes a `dlt` source you hand straight to `cognee.remember(...)`, so it reuses cognee's
DLT ingestion path (`resolve_dlt_sources` → `ingest_dlt_source` → `orphan_cleanup`). One
message is one document: the subject is its title and the headers, labels and plain-text body
are its content. That gives you **incremental re-sync** (Gmail's `historyId` cursor, `merge`
write disposition) and **forget-on-delete** (messages you delete or trash in Gmail are removed
from memory on the next sync) without a parallel ingestion path. The package also carries the
provider's connection contract (`CONNECTOR`: the Google OAuth flow, token refresh and
revocation, the label picker and the sync scope) for a cognee server or Cognee Cloud, and a
`cognee_host/` adapter that plugs it into a cognee server's `/api/v1/integrations/gmail/*`
routes.

## Install

```bash
pip install cognee-connector-gmail             # the connector and its DLT source
pip install "cognee-connector-gmail[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

The source needs a cognee whose ingestion scopes DLT document cleanup by table
(`DOCUMENT_SYNC_VERSION >= 1`, cognee 1.6.5 or later); an older host is refused with a clear
message when the source is built. `cognee-connector-google`, the OAuth client and contract the
three Google connectors share, is installed with it.

## Configuration

Every environment variable the settings class
(`cognee_connector_gmail.settings.GoogleGmailSettings`) reads, with the `GOOGLE_GMAIL_` prefix.
Names are unchanged from cognee. The SDK source reads none of them: it takes the OAuth client
secret and token files as arguments (see Setup).

```bash
# Gmail: one OAuth client in a Google Cloud project, consented per account.
# Personal and Workspace accounts use the same client and the same flow; they
# are told apart by the hosted-domain claim after connecting. Its own OAuth
# client and consent screen, separate from Drive's and Calendar's: Google's
# revoke endpoint kills a whole grant, so one product's disconnect must never
# take another's grant with it.
# CLIENT_ID / CLIENT_SECRET: from the OAuth 2.0 Client ID (type: Web
# application) in that project's Credentials page.
# STATE_SECRET: any long random string. It signs the OAuth state parameter,
# which is the only thing binding a callback to the user who started it.
# REDIRECT_URI: must be listed verbatim as an authorized redirect URI on the
# client, and point at this server's callback.
# FRONTEND_BASE_URL: where the browser lands after connect/cancel/error
# (appends ?gmail=<outcome> to /integrations).
# Mail is opt-in: nothing syncs until the user selects labels in the
# integrations page.
# Note: gmail.readonly is a restricted scope, so a public launch needs
# Google's OAuth verification first. Test users work without it.
#GOOGLE_GMAIL_CLIENT_ID=""
#GOOGLE_GMAIL_CLIENT_SECRET=""
#GOOGLE_GMAIL_STATE_SECRET=""
#GOOGLE_GMAIL_REDIRECT_URI="http://localhost:8000/api/v1/integrations/gmail/callback"
#GOOGLE_GMAIL_FRONTEND_BASE_URL="http://localhost:3000"
```

| Variable | Read by | Purpose |
| --- | --- | --- |
| `GOOGLE_GMAIL_CLIENT_ID` | settings | The OAuth client id of the Gmail consent screen |
| `GOOGLE_GMAIL_CLIENT_SECRET` | settings | Its client secret |
| `GOOGLE_GMAIL_REDIRECT_URI` | settings | The callback URL, listed verbatim on the client |
| `GOOGLE_GMAIL_FRONTEND_BASE_URL` | settings | Where the browser lands after the OAuth round-trip |
| `GOOGLE_GMAIL_STATE_SECRET` | settings | Signs the OAuth `state` parameter (cognee host) |

## Usage

```python
import cognee
from cognee_connector_gmail import gmail_source

await cognee.remember(
    gmail_source(label_ids=["INBOX"]),
    dataset_name="gmail_inbox",
    primary_key="id",
    write_disposition="merge",  # REQUIRED — see note below
    max_rows_per_table=0,  # unlimited: orphan-cleanup sees the whole corpus
)

answer = await cognee.search(
    query_text="Summarize the most important emails in my inbox.",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["gmail_inbox"],
)
```

Re-running `remember(...)` with the same dataset syncs only the delta and forgets any mail
removed from Gmail. See `examples/gmail.py` for the full demo.

Deletion propagation requires a successful foreground sync (the default).
`cognee.add(..., run_in_background=True)` skips orphan cleanup to preserve existing memory if
the detached ingestion fails. To purge deletions already staged by a background run, the
affected source tables must be loaded again in the foreground; a no-change incremental retry
alone does not guarantee reconciliation.

Changing `label_ids` starts a backfill for the new selection, including messages that predate
the previous sync. Reordering the same labels keeps the incremental cursor. Changing the
selection does not purge previously indexed mail; use `cognee.forget(dataset="gmail_inbox")`
to clear the dataset before replacing its scope.

> **`write_disposition="merge"` is mandatory.** The add pipeline defaults to `"replace"`
> (drop + reload each run), which would wipe the whole synced inbox on the second sync.

`gmail_source` takes, besides `label_ids`: `credentials_path` and `token_path` (the OAuth
client secret and the cached user token, `credentials.json` and `token.json` next to the
script by default), `max_results` (cap a backfill, for demos; no cursor is saved then),
`quota_units_per_minute` (the share of Gmail's per-user quota this mailbox may use, 5,000 by
default), `resource_name` (give each account its own name when several sync into one dataset,
so their cursors and quota budgets stay separate), `check_active` (a host's authorization
checkpoint) and `service` (a pre-built Gmail client; `build_gmail_service_from_access_token`
makes one from a token a host already holds).

## Setup (OAuth)

1. In Google Cloud Console: enable the Gmail API, configure an OAuth consent screen (add
   yourself as a test user), and create an **OAuth 2.0 Client ID → Desktop app**. Download
   the client-secret JSON.
2. Save it as `credentials.json` (or pass `credentials_path=...`). The first run opens a
   browser to consent and caches a token at `token.json` (written `0600`).
3. Set your `LLM_API_KEY` like any other cognee run.

Access is **read-only** (`gmail.readonly`) and strictly opt-in: nothing is fetched until you
construct a source and call `remember`. Scope what leaves your mailbox with `label_ids`, keep
`token.json` private, and use a dedicated dataset so you can wipe it with a single
`cognee.forget`.

## Running on a cognee server

```python
from cognee_connector_gmail.cognee_host import register

register()  # before the app starts; the provider appears in GET /api/v1/integrations/status
```

`routers()` in the same module is empty: Gmail mounts no routes of its own. The OAuth flow
runs on the generic `/api/v1/integrations/gmail/...` routes, and the server receives no
webhooks from Gmail; a connection syncs on the user's Refresh action.

On a cognee server the integration does three things:

- **Connects a Google account.** The consent asks for `openid email` and `gmail.readonly`
  only, with `access_type=offline` and `prompt=consent` so Google issues a refresh token, and
  without `include_granted_scopes`, so this grant is never folded together with a Drive or
  Calendar one. The credential is keyed on the account's stable Google subject; its email,
  hosted domain and account type (`workspace` or `personal`) are kept as cleartext metadata.
- **Keeps mail opt-in.** A new connection starts with an empty label selection
  (`selected_label_ids: []`) and syncs nothing until the user picks labels with
  `GET`/`PUT /api/v1/integrations/gmail/resources`; a null selection means the whole mailbox.
  Each sync hands the exact selection to `gmail_source` (`label_ids`) under one DLT resource
  per account, into the dataset `gmail_<account email>_<hash>`, named as cognee always named
  it, and retires stale tables only after a clean run.
- **Rotates and revokes tokens.** The access token is refreshed five minutes before it
  expires; Google issues no new refresh token, so the stored one is carried forward. An
  `invalid_grant` answer means the account revoked access on Google's side and revokes the
  local credential, since no webhook would ever say so. Disconnecting revokes the grant at
  Google, best-effort.

## What is synced

| Row | Id | Title | Content |
| --- | --- | --- | --- |
| Message | the Gmail message id | the `Subject` header | `From`, `To`, `Cc`, `Date`, `Received` (Gmail's `internalDate`), `Labels` and `Thread` lines, then the first `text/plain` part of the MIME tree |

- Only the columns cognee reads are kept: `id` (the merge key), `title`, `content` and the
  `_deleted` hard-delete marker. Attachments are not read. An HTML-only message has no
  plain-text body and falls back to Gmail's snippet, so the graph never sees raw markup.
- The first run does a full, label-scoped backfill and records the mailbox `historyId` before
  listing, so nothing that changes during a slow backfill is missed. Later runs read
  `users.history.list` and emit only the delta: added or relabelled messages are fetched again,
  deleted or trashed ones become `_deleted` tombstones. A message that left the selected
  labels (trashing INBOX mail fires `labelsRemoved`, not `messagesDeleted`) is forgotten, not
  re-ingested. An expired history id (Gmail keeps about a week) falls back to a full backfill
  that reconciles the same way.
- A message Gmail reports as gone (404/410) counts as `skipped_unavailable`; a transient
  failure (5xx, rate limit, network) is re-raised instead, so the cursor never advances past a
  message that could be fetched later. `cognee_sync_stats` holds counts only: `scanned`,
  `skipped`, `failed`, `deleted`, plus `skipped_unavailable` and `failed_message_fetch`.

## Quota

Every call is paced against Gmail's per-user quota (6,000 units per minute; fetching one
message costs 20), so a backfill runs at roughly 250 messages a minute with the default
budget of 5,000 units. Sources with the same `resource_name` share one budget in a process,
so a re-triggered sync that overlaps a running one cannot together exceed it. When Google
still answers `rateLimitExceeded` after the client library's own retries, the sync waits out
up to five one-minute windows before failing: the quota is shared by every client of the
same account under the same Cloud project, which local pacing cannot see.

## The connection contract

`cognee_connector_gmail.CONNECTOR` is Gmail on the `cognee-connectors-core` contract, for a
host that drives connectors generically (Cognee Cloud, a future cognee release): the shared
Google descriptor (`cognee_connector_google.google_descriptor`) with Gmail's read-only scope,
on `GoogleConnector` for the userinfo lookup that keys the credential on the Google subject
(tier 2), plus the label picker (`list_resources`) and the sync scope (`sync_scope`). With no
resources named, the scope is the whole mailbox (one item, `mailbox`, with `label_ids` null
for the source); with resources, the labels named. A host that keeps a selection narrows
before calling, as the contract has it. The connector refreshes through the refresh grant,
keeps the stored refresh token Google does not rotate, revokes the grant on disconnect without
ever raising, and polls hourly for a host that schedules syncs. Its dataset is
`gmail_<google subject>`, as Cognee Cloud names it; the cognee server keeps its own
email-based name (see above).

## Privacy

This connector reads the content of your email. It is **opt-in**: nothing is fetched until
you explicitly construct a source and call `remember`, or connect an account to a server and
pick labels. Use `label_ids` to scope what leaves your mailbox, keep the OAuth token file
private, and prefer a dedicated dataset so you can `cognee.forget` the inbox in one call.
Logs and `cognee_sync_stats` carry counts and HTTP statuses, never subjects or bodies.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/google:connectors/gmail python -m pytest connectors/gmail/tests -q
PYTHONPATH=core:connectors/google:connectors/gmail python scripts/check_cognee_free.py cognee_connector_gmail
```

`tests/unit` covers the source against a fully mocked Gmail API (`FakeGmailService`, no
Google client libraries or live credentials needed), including an offline end-to-end run
through a real `dlt` merge that proves a delete marker physically removes the row; the
contract object with Google's token and userinfo endpoints faked; and the cognee adapter and
sync, which need cognee installed.
