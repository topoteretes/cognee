# Import Slack conversations into Cognee

The SDK's Slack integration can import selected channel history or complete
threads into an existing Cognee dataset. It uses native `add`, `update`,
`cognify`, and `datasets.delete_data`; the connector stores only selection and
run status alongside the existing encrypted Slack connection.

## Install or upgrade the Slack app

1. Configure the existing Cognee Slack OAuth connection. Add these bot scopes:
   `channels:history`, `groups:history`, and `groups:read`, alongside the existing
   `commands`, `chat:write`, `im:write`, and `channels:read` scopes. Reinstall or
   reconnect the app to grant the new scopes. Reading history is opt-in; granting
   scopes does not start an import.
2. Invite the app to each selected channel. The Slack person who connected the
   workspace must also be a member. Bulk imports are managed by the connection
   owner; linking another member's Cognee account does not grant them the
   installation's ability to export channel history.
3. Add `/cognee-import` with the same request URL as `/cognee-remember`:
   `https://YOUR_API/api/v1/slack/commands`.
4. Add a **message shortcut** named **Remember entire thread**, callback ID
   `remember_thread`. Set the Interactivity request URL and the **Options Load
   URL** to `https://YOUR_API/api/v1/slack/interactive`. The options URL supplies
   destination datasets from the caller's actual Cognee permissions.
5. Restart the updated API. Scheduled refreshes run inside the API process once
   a selection is enabled. `SLACK_HISTORY_SYNC_ENABLED=false` disables this
   worker for deployments that use an external scheduler instead.

The existing `/cognee-remember` command and single-message shortcut keep their
behavior. `/cognee-ask` already searches imported memory using Cognee's dataset
permissions. No frontend or MCP changes are required.

## Select information in Slack

Run `/cognee-import` (seven days by default), or `/cognee-import 14`. The dialog
lets you choose a destination dataset, channels, thread links, and ongoing sync.
The message shortcut starts the same dialog with that entire thread selected.

The dataset picker offers datasets where the owner has **read, write, and
delete** permission: native document replacement and source deletion require
those rights. The importer rechecks permissions before fetching and before
writing. A channel allowlist also applies to the *selected source channels*,
not only the channel where a command is invoked.

By default, dates select **threads started during the period**. The optional
**active during the period** mode scans older roots too, so a recent reply to an
old thread is included. This requires more Slack calls. In both modes the
selected conversation includes all currently available replies, even those
outside the date range. Thread links select complete conversations independently
of channel dates. Files/attachments and DMs are not imported; message text,
authors, timestamps and source links are retained. Cognee bot replies are skipped
to avoid feeding its answers back into its own source memory.

Confirmation is private and sent after indexing completes. Large imports may
outlive Slack's response URL or a server restart; use the status endpoint below
and retry an interrupted selection. Repeating an import does not duplicate its
documents.

## Dataset and node-set organization

Use datasets as access boundaries and node sets to organize channels. Each
conversation receives `slack`, `slack:<team_id>`, and
`slack:channel:<channel_id>` node sets. Node sets do not enforce permissions.
Choose an appropriately restricted destination for private channels; source
membership is not copied into Cognee ACLs and this feature never grants access.

The importer manages only documents carrying its own source stamp in the target
dataset. Manually remembered Slack notes, other sources, and other datasets are
not swept by source reconciliation.

## API and SDK

All HTTP endpoints require a Cognee bearer token/session belonging to the Slack
connection owner. JSON fields use snake_case.

```text
POST /api/v1/slack/history/{team_id}/import
PUT  /api/v1/slack/history/{team_id}/sync
GET  /api/v1/slack/history/{team_id}
```

Example import request:

```json
{
  "dataset_id": "YOUR_DATASET_UUID",
  "channel_ids": ["C0123456789"],
  "days": 14,
  "thread_mode": "active",
  "thread_links": []
}
```

Instead of `days`, use timezone-aware `oldest` and optional `latest` dates, for
example `2026-09-01T00:00:00Z`. For selected conversations alone, omit
`channel_ids` and dates and supply `thread_links`, or structured
`threads: [{"channel_id": "C0123456789", "ts": "1700000000.000001"}]`.

To configure periodic refresh, PUT:

```json
{
  "enabled": true,
  "interval_seconds": 21600,
  "selection": {
    "dataset_id": "YOUR_DATASET_UUID",
    "channel_ids": ["C0123456789"],
    "days": 14
  }
}
```

A saved selection resolves `days` into a **fixed start**, then extends to the
present on each refresh. There is one selection per workspace and destination
dataset; saving it again replaces that selection. Separate destination datasets
can have different source selections. To disable one, PUT the same selection
with `enabled: false`. Disabling stops future runs, not an already running import.
GET returns saved selections and the last run's status/counts, never tokens or
message content. Failed refreshes become eligible for retry after ten minutes;
successful refreshes use the configured interval (six hours by default).

For an embedded SDK application, the same entry points are:

```python
from cognee.modules.integrations.slack.history import import_slack_history
from cognee.modules.integrations.slack.history_models import SlackHistoryRequest

result = await import_slack_history(
    team_id,
    SlackHistoryRequest(dataset_id=dataset_id, channel_ids=channel_ids, days=14),
    user=owner,
)
```

Use the HTTP API when Cognee already runs in another process. The runnable
[HTTP example](../examples/python/integrations/slack_history.py) avoids opening
the server's embedded databases in a second process.

## Sync guarantees and limits

- Every channel/thread page is fetched before native writes begin. API errors,
  missing pagination cursors, and request/message caps fail the import rather
  than marking a partial snapshot complete. Defaults are 1,000 requests and
  50,000 messages; API callers can explicitly raise these bounded limits.
- Stable identity includes destination dataset, Slack workspace, channel, and
  root timestamp. Changed threads go through native `update`, so stale derived
  graph/vector content is removed. Identical imports reuse the same documents.
- Apps using rotating Slack tokens refresh expiring credentials before history
  reads. Rotation preserves source selections, channel restrictions and account
  ownership; it cannot reactivate a concurrently revoked connection.
- Scheduled reconciliation also re-fetches previously imported conversations
  in the selected channels, catching edits, deleted replies, late replies, and
  changes made while the API was offline. A confirmed missing thread deletes
  only its corresponding source document through native deletion. Lost scopes,
  channel access, and transient errors never stand in for deletion.
- A rolling date window is **not retention**. A shorter import never deletes
  older documents just because they are absent from that window. Reconciliation
  reflects what Slack currently makes available; it cannot recover history
  already removed by Slack retention policies.
- Native multi-document updates are retryable, not atomic across the complete
  import. If indexing fails after ingestion, retrying still runs indexing even
  when raw content is unchanged. One-off imports interrupted by process exit
  require a retry; enabled scheduled selections recover on a later tick.
- Run one API worker with local embedded stores. Native dataset locks and this
  refresh worker are process-local; a multi-worker deployment must designate a
  single history worker and serialize external import requests.

Slack API references: [history](https://docs.slack.dev/reference/methods/conversations.history/),
[threads](https://docs.slack.dev/reference/methods/conversations.replies/),
[rate limits](https://docs.slack.dev/apis/web-api/rate-limits/). Internal customer
apps retain the normal history tier; stricter limits apply to some commercially
distributed apps. The connector honors `Retry-After` and bounded retries.
