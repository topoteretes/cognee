# Slack SDK connector

A Slack data-source connector for [cognee](https://github.com/topoteretes/cognee): sync the
conversations of a channel into memory, incrementally, and forget a conversation when it is
deleted in Slack.

It exposes a `dlt` source you hand to `cognee.remember(...)`. One source is one channel of one
workspace. A thread (a message with replies) is one **normal document**, and the messages
without replies of one UTC day are one document for that day, so it flows through chunking and
LLM entity extraction like any other text.

## Requirements

Bundled with the Cognee SDK; no extra is needed (the client uses `httpx`, a core dependency).
A core that does not read per-row structure (`DOCUMENT_SYNC_VERSION` below 3) is refused.

The bot token needs these scopes:

| Scope | Used for |
|---|---|
| `channels:history` / `groups:history` | `conversations.history` and `conversations.replies` of a public / private channel |
| `channels:read` / `groups:read` | `conversations.info` (the channel name) and, in the integration, the channel picker and the membership check |
| `users:read` | `users.info`: authors and mentions are rendered by name |

The app must be a member of the channel. Workspaces connected through the Slack integration
before `users:read` was added have to reconnect once (see "Limits").

## Setup

**With a bot token you already hold (the SDK, below):**

1. Create a Slack app with the bot scopes above and install it into the workspace.
2. Invite the app to each channel you want to sync (`/invite @<your app's bot name>`).
3. Pass the bot token, the workspace id (`team_id`), the channel id and a `resource_name` of
   the channel's own (it names the staging table and the sync state). The connector never
   refreshes a token: a host hands over a fresh one.

**Through the Slack integration on your own server (connect, pick channels, `@cognee`):**

1. The server must be reachable from Slack over public HTTPS (a domain, or a tunnel such as
   ngrok for testing): Slack calls the events URL, the interactivity URL and both OAuth
   redirect URLs. The UI must call the API on that same host, because the OAuth callback
   checks a cookie set when the UI started the install; on another host it ends in
   `error_invalid_state`. Locally, start the UI with the API proxy and tunnel the UI, not
   the API:
   ```bash
   cd cognee-frontend && COGNEE_DEV_API_PROXY=http://localhost:8000 npm run dev
   ngrok http 3000
   ```
   (with `cognee-cli -ui`, put `COGNEE_DEV_API_PROXY=http://localhost:8000` in the `.env`
   instead: the CLI starts the UI with that environment), then open the UI at
   `https://<ngrok>` (not `localhost:3000`, which the CLI opens by itself) and set
   `SLACK_FRONTEND_BASE_URL=https://<ngrok>`. The UI forwards `/api/v1` to the API, Slack's
   signed requests included.
2. Create the app from `scripts/slack-app-manifest.sdk.yml` (api.slack.com/apps > Create New
   App > From a manifest), after replacing `<ngrok>` with your host and, for a real
   deployment, the app's name and bot name. Slack tries to verify the events URL when the
   manifest is saved; that first check fails until step 4, which is expected.
3. Set in the server's `.env` (`.env.template`, "Slack"):
   - `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`, `SLACK_SIGNING_SECRET` from the app's Basic
     Information (the events URL answers every request, Slack's check included, only once
     the signing secret is set);
   - `SLACK_REDIRECT_URI` (`https://<host>/api/v1/integrations/slack/callback`) and
     `SLACK_MEMBER_REDIRECT_URI` (`https://<host>/api/v1/integrations/slack_member/callback`),
     byte for byte as in the manifest;
   - `SLACK_FRONTEND_BASE_URL`, where the browser returns after connecting (the callback
     answers 503 without it);
   - `INTEGRATION_CREDENTIALS_KEYS` (or `INTEGRATION_CREDENTIALS_KEY`), the key the bot token
     is encrypted with at rest: without it the connect fails after Slack's consent page, and
     the cause is only in the server log.
4. Start (or restart) the server with one worker (the default `entrypoint.sh` uses `-w 1`: a
   sync's lock and its run exclusivity are per process, so two workers could sync one
   workspace at once), then in the app's Event Subscriptions page retry the Request URL
   verification.
5. Connect the workspace (`POST /api/v1/integrations/slack/authorize`, then open the returned
   URL in the browser where you are logged in to Cognee), invite the app to each channel
   (`/invite @<bot name>`, public channels included) and pick them (`GET`, then `PUT
   /api/v1/integrations/slack/resources`). Members link
   their own Cognee account with `@cognee link`.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import slack_source

await cognee.remember(
    slack_source(
        access_token="xoxb-...",
        team_id="T0123ABCD",
        channel_id="C0456EFGH",
        resource_name="slack_channel_general",  # one per channel, never shared
    ),
    dataset_name="slack_acme",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; forget-on-delete must see the whole channel
    self_improvement=False,
)

answer = await cognee.recall(
    "What did we decide about the launch?",
    datasets=["slack_acme"],
    node_name=["slack:channel:C0456EFGH"],  # optional: only this channel
)
```

Run the same call again to sync. A runnable script is in `examples/guides/slack.py`.

### Through the Slack integration

A self-hosted server with the Slack app configured (`SLACK_CLIENT_ID` and friends in
`.env`) syncs through the generic integration routes:

- `GET /api/v1/integrations/slack/resources` lists the channels the app is a member of. The
  command allowlist (`PUT /api/v1/slack/channels`) never narrows it: the allowlist decides
  where Cognee's commands and "Remember this" work, never what is synced. A private channel is
  listed only when the Slack user who connected the workspace is a member too, so the picker
  never names a private channel that person cannot see. A private channel whose members cannot be read right now is
  left out, and only their count is logged, unless it is selected: then it stays listed, so a client saving back the
  listed ids does not deselect it (the sync checks membership again before it reads it). A rate limit is never waited
  out inside the request: when Slack limits the channel list, the route answers 429 with Slack's `Retry-After`, and
  the allowlist editor (`GET`/`PUT /api/v1/slack/channels`) does the same. Slack's channel list leaves archived channels out, so a
  selected channel Slack has archived since is added to the list, with `attributes.is_archived`
  true, named under the same rule; a client that saves back the listed ids then keeps it selected.
  `PUT` the ids to sync. A new connection selects nothing, and nothing syncs until channels are
  picked.
- `POST /api/v1/integrations/slack/sync` starts a sync now. An in-process tick also syncs every
  connection with a selection every six hours, and ten minutes after a run that was cut short by
  the request budget, a rate limit or a Slack outage; `SLACK_HISTORY_SYNC_ENABLED=false` turns it
  off. Each due workspace starts at once in its own task, but only one Slack sync, scheduled or
  "Sync now", runs per process at a time; the others wait, in order, reported as
  `waiting_for_lock`, and read their selection afresh when their turn comes.
- Each channel takes a fresh token when its walk starts, so a rotating token (12 hours) stays
  valid through a long multi-channel sync.
- All channels of a workspace go into one dataset, `slack_account_<hash of the team id>`.
  Saving a changed selection starts a sync at once (or one more run of a sync already running),
  so deselecting a channel removes its documents right away unless a failure blocks the
  retirement; deselecting every channel included.

Before reading a channel the sync asks Slack about it and follows Slack's own permissions, because
the Slack user who connected the workspace is the reader of the dataset:

| Slack's answer | What the sync does |
|---|---|
| the app is in the channel; for a private channel the connecting user is too | syncs it; for a public channel the connecting user need not be a member, since any workspace member can read it |
| the channel is archived | keeps its documents without reading the channel, counted under `channels_archived` (not a failure); it syncs again once it is unarchived. Slack keeps an archived channel's history and its members still read it. A private one the connecting user is not in is removed, as in the next row |
| a private channel the connecting user is no longer in | removes its documents like a deselected channel's, counted under `channels_retired` |
| the channel was deleted, or the app was removed from it | keeps its documents and reports it on every sync under `failed_channel_unavailable`, until it is deselected; with the setting `forget_unavailable_channels` on (`PUT /api/v1/slack/settings`, owner only) its documents are removed instead, counted under `channels_retired` |
| a rate limit, or a member list Slack does not return | keeps its documents, counts `failed_rate_limit` or `failed_channel_access`, and removes nothing from any channel on this sync |

Deselected and retired channels are removed after every sync, unless a channel failed for a
reason other than being unavailable before anything of it was stored: such a failure could leave
a channel whose old table is the only copy of its data, so the removal waits for the next sync.
A channel whose own walk was cut short (the request budget, a rate limit, an outage after rows)
was stored and resumes on the next sync; it holds only its own table, so it does not hold the
removal back, which under Slack's Tier 1 limits would otherwise almost never run.

When Slack answers the sync's first call with `account_inactive`, which for a bot token means the
workspace or the app's bot user was deleted, the connection is ended as an uninstall ends it
(revoked by the provider) and the tick stops syncing it. Its documents stay until the owner
deletes them. Any other token error (`invalid_auth` can be an IP allowlist) fails the sync and
leaves the connection as it is.

When the app is uninstalled (`app_uninstalled`), the connection is revoked and, as Slack's
Developer Policy asks, the workspace's data is deleted 14 days later: the documents the
connector synced into the workspace's dataset (every owner's copy, a previous owner's who
disconnected keeping it included; a dataset is removed only when that leaves it empty, so
anything else in a dataset of that name stays), its dlt
staging database, state and completed load packages (`drop_dlt_dataset`), and every member's
"Remember this" and `@cognee remember` items from that workspace
(`sync.forget_uninstalled_workspaces`, run by the same in-process worker as the sync tick, every
60 seconds, also with `SLACK_HISTORY_SYNC_ENABLED=false`; a host with its own scheduler calls it).
Connecting the workspace again within the 14 days cancels the deletion. `tokens_revoked` alone, a
disconnect and `account_inactive` delete nothing. Slack delivers events best effort: an
`app_uninstalled` that never arrives starts no deletion. The owner's connection status says it
(`GET /api/v1/integrations/slack/connection`: `dataDeletedAfter`, then `dataDeletedAt` once the
deletion ran), also to an owner who had disconnected keeping the data; a previous owner whose
connection another user took over is not told (G84).

### Talking to Cognee from Slack

Members use `@cognee`, in a channel the app is in or in a direct message to the app
(`handle_cognee_mention.py`; the app also needs `app_mentions:read`, `chat:write`,
and `im:history`, and the events `app_mention` and `message.im`, all in
`scripts/slack-app-manifest.sdk.yml`; a workspace connected before has to reconnect):

| Message | What it does |
|---|---|
| `@cognee <question>` | in a channel, answers in the question's thread (or the thread it was asked in) with a message everyone in the channel sees (`chat.postMessage`), so it reads only what all of them can: the workspace's dataset within that channel, when it is synced, and the selected public channels (Slack's `users.conversations` for the app, asked on every question, so a channel made private drops out at once), never a selected private channel other than this one and never the asker's other datasets. It reads those channels' messages alone (a chunk search, `RAG_COMPLETION`): one entity is one node for every channel that names it, so the hybrid entity lane would carry a relation from a channel outside the scope. A channel that is not synced, with no selected public channel, reads nothing and says so. In a channel shared beyond this workspace (`conversations.info`'s `is_ext_shared`, Slack Connect, or `is_org_shared` on Enterprise Grid) it reads that channel alone, since members from elsewhere cannot read this workspace's channels; a guest asking (`users.info`) gets that channel alone too; when Slack cannot say, it reads that channel alone too and the asker is told. In a direct message it answers with a message in the direct message, from the member's memory: the workspace's dataset within the selected channels Slack says the member is in (`users.conversations`; when Slack cannot say, without it, and the member is told) and their other datasets whole. Either way the answer is the reply itself (no header), shown below; a miss or a failure is shown to the asker alone. A question Slack sends again because the ack was slow (`X-Slack-Retry-Reason: http_timeout`) is already being answered and is not answered twice; one sent again after a failed delivery is answered |
| `@cognee remember <text>` | saves a note Slack never saw, e.g. a decision from a call; the note is the item of that message, so `@cognee forget` in its thread, *Forget this* on it, or deleting the message forgets it |
| `@cognee remember` in a thread | saves the thread's first message, followed like "Remember this": an edit or a deletion in Slack forgets the copy |
| `@cognee forget` in a thread (a channel or a DM) | forgets the member's own copy or note of the thread's first message; the *Forget this* shortcut does it for any message. Other members' copies stay, and the channel allowlist never blocks it. On a message of a synced channel, its author or the connection's owner also takes it out of the workspace's memory (below) |
| `@cognee link` | the page where the member links their Slack account |
| `@cognee unlink` | ends the member's own link |
| `@cognee status` | which Cognee account the member is linked to, who connected Slack, whether this channel is synced and whether commands work here, what an answer here reads, the workspace's memory, and the last and next sync; a row that needs Slack and cannot reach it says Unknown |
| `@cognee help`, or a bare mention | what the app does |

An answer is a message; every other reply is ephemeral, shown to the member who asked. A
question that should stay private goes in a direct message. A direct message is always allowed; in a channel the command
allowlist applies, except to `forget`, `unlink` and `status`. The "Remember this" and "Forget this" message shortcuts save and forget
any message. There are no slash commands.

An answer reads like cognee's own Search answer. The LLM's Markdown is rendered as Slack
mrkdwn by one renderer (headings become a bold line, a leading "Answer" heading goes, bold,
italics, strikethrough, bullets and https links are converted, code is kept; everything stays
escaped, so an answer cannot ping a channel or disguise a link); the notification text is the
same answer as plain text. Below each fact one line numbers the documents it read, `Sources ·
N   1 #general · ana · Nov 5   2 …`, each a link to the message in Slack: the channel is the
name in the row's structure (`document_parent`), the author and day are read from the row's
title, so neither changes what a row is and nothing is re-cognified. A line then says what the
answer read: `From #channel and N public channels`, `From #channel` in a Slack Connect channel
or when Slack could not say which channels are public, `From your channels and your memory` in
a direct message.

**Rating.** Under every answer, 👍 and 👎. Each answer is recorded as session Q&A (session
`slack:<team>:<member>`) under the person who connected Slack, who owns the workspace dataset,
with the graph nodes and edges its workspace search read; a channel answer keeps its question
and answer text, a direct message's keeps neither (only the graph ids), since it may quote the
member's own datasets. Only the person who asked can rate: the click's Slack user (signed by
Slack) must own the answer's session, and the button carries the answer's random id alone, so
anyone else's click changes nothing and they are told so privately. 👍 records 5 and 👎 1 as
the answer's `feedback_score`; a second click rates again and replaces the first rating. Each
rating runs improve's `feedback_weights` pipeline at once over that session and the workspace
dataset, moving the `feedback_weight` of the graph elements the answer used (elements of the
member's own datasets are not in the workspace graph and stay untouched). **Ratings change
answers only once `DEFAULT_FEEDBACK_INFLUENCE` is above 0** (default `0.0`, a separate SDK
setting); until then they are collected and nothing reads them. With `CACHING=false`, or when
the cache cannot keep the record, the answer is posted without the buttons. The cache's own
expiry applies. The records quote the workspace, so the 14-day deletion after an uninstall
deletes every member's, and a disconnect that deletes the data deletes those kept under the
disconnecting user.

A mention is answered once per delivery. The event is acknowledged at once and answered in the
background, so Slack sends it again only when the acknowledgement did not reach it in time (a
restart or a network failure), and that copy is answered too: the member sees the reply twice,
and a question is searched twice. Nothing is stored twice: a note's and a saved message's ids
derive from the member and the message, and a second `forget` finds
nothing to forget. The event's `event_id` is not tracked, so no table of seen events grows
(Goran, 2026-11-10).

A member's questions and saved messages use the memory of the Cognee account their Slack
account is linked to. The member links it themself with Sign in with Slack: logged in to
Cognee, they open the link `@cognee link` sends, the server's
`GET /api/v1/integrations/slack_member/start` (beside `SLACK_MEMBER_REDIRECT_URI`), which sets
the nonce cookie and sends the browser to Sign in with Slack (a client of its own can call
`POST .../slack_member/authorize` and follow the URL instead); Slack proves which member signed in, and the browser's nonce cookie that
it is the browser that started it, so nothing typed into Slack or forwarded to someone else can
link an account. `DELETE /api/v1/integrations/slack_member/connection` unlinks it. The app needs
the redirect URL `SLACK_MEMBER_REDIRECT_URI` (`.../api/v1/integrations/slack_member/callback`);
only `openid` is requested and Slack's user token is not kept. Until a member links, only the
Slack user who installed the app reaches the installer's memory.

## What is ingested

Two kinds of rows. A thread (a top-level message with replies) is one row with every reply;
the messages without replies are one row per UTC day per channel (G7), so a channel where
people rarely open threads becomes a few documents a day, not one per message:

| Column | Value |
|---|---|
| `id` | `<channel_id>:<root_ts>` for a thread, `<channel_id>:day:<YYYY-MM-DD>` (UTC) for a day (permanent: the dlt primary key) |
| `title` | `<the first ten words of the row's first message> — <author>, <YYYY-MM-DD>` |
| `content` | one block per message, oldest first: `[<time> UTC] <author>: <text>`, then `Attached: <file names>`; the text is the message's text with Slack markup decoded, then the text of its attachments (a shared message, a bot's alert), or of its blocks when there is no other text |
| `url` | the permalink of the row's first message, `https://<workspace>.slack.com/archives/<channel>/p<ts>` |
| `cognee_node_set` | `["slack:channel:<channel_id>"]`, plus `"slack:shared"` for a channel shared with another organization (Slack Connect) |
| `cognee_structure` | `{"ancestors": [{"kind": "slack_channel", "id": <channel_id>, "name": "#<channel>"}]}` |

- Authors and `<@U…>` mentions are rendered by full name (`real_name`, falling back to the
  display name), looked up with `users.info` once per person and run: a full name changes and
  repeats less often than a display name. A user Slack no longer
  returns (deleted, or not visible to the app) is rendered by id. A bot message without a user
  is rendered by its bot name.
- Slack markup is decoded: `<#C…|ops>` becomes `#ops`, `<https://…|label>` becomes
  `label (https://…)`, `<!here>` becomes `@here`, a user group its handle, and `&amp;`, `&lt;`,
  `&gt;` their characters. A link preview Slack unfurled is left out (its URL is in the text).
- A channel shared with another organization (Slack Connect, `is_ext_shared`) is synced like
  any other: the partner's messages reach memory and the configured LLM. Its documents carry the
  node set `slack:shared`, so a search can leave them out, and the picker marks it
  `attributes.is_shared`.
- Files are named, never downloaded.
- Left out: messages to the Cognee app, any message that starts by mentioning it, a command
  (`@cognee forget`) or a question (`@cognee what did we decide?`): they talk to the app, not
  to the channel, and a `remember` note is stored on its own. A message that only mentions
  the app later is content. The same holds for any other app (a user `users.info` marks
  `is_bot`, such as another Cognee app): a message that opens by mentioning it is a command
  to it, and in a thread opened by such a command, what apps reply is left out too, while
  what people reply stays. Other apps' messages outside such threads (CI results, alerts)
  are content.
- Left out: messages that are not something a person wrote (joins, leaves, topic and purpose
  changes, pins, deleted-message placeholders and every other non-content subtype) and the
  Cognee app's own messages. A thread or a day with nothing left produces no row.
- A reply also sent to the channel stays in its thread; it is not a document of its own.
- A message that gets its first reply leaves its day's document and becomes a thread of its
  own; a thread that loses its last reply joins its day's document again.
- The channel name is not in the text, only in the structure, so renaming a channel
  re-cognifies nothing.

## How sync and forget-on-delete work

**Incremental model: a fingerprint re-walk, with no time window.** Every run reads
`conversations.history` newest first, paged with `latest`. Each root keeps a fingerprint in the
dlt state: `(reply_count, latest_reply, edited.ts, subtype)`. A root whose fingerprint is
unchanged costs nothing beyond its share of the history page; for a changed or new root the
thread is read with `conversations.replies` and rendered again. A new, changed or gone message
without replies marks its day, and the day's document is rendered again from that day's messages
once the walk has read the whole day. A channel with no changes yields no rows, so cognify does
nothing.

State (`dlt.current.resource_state()`): `{"v": 1, "roots": {"<root_ts>": "<fingerprint>"},
"days": {"<YYYY-MM-DD>": 1}?, "dirty_days": ["<YYYY-MM-DD>", ...]?, "position": "<ts>"?,
"missing": ["<root_ts>", ...]?, "names": {"<user_id>": "<name>"}?, "newer": ["<day start>",
"<ts>"]?}`. `roots` holds every
top-level message the walk read, a message without replies included (it is part of its day's
row), and, marked with a leading `~`, every root whose replies were read and rendered nothing
(the app's own thread, a join with replies), so that thread is not read again until it changes.
`days` holds the days that have a row and `dirty_days` the days marked and not yet rendered.
While a walk is unfinished, `position` is the end of the oldest day it reached, so the next run
reads that day again from its last message, `missing` the known roots its pages showed to be
gone, and `names` the authors' names it looked up, so a walk over many runs asks Slack for each
author once. While a walk is unfinished each run first reads what is newer: from the newest
known root itself (one request in a quiet channel), and from the start of any day marked above
the walk, including the newest known root's day when a new message arrived in it, never below
the walk's own position, rendering each day as soon as it has read it whole; `newer` keeps where that read started and how far it
reached, so a run cut inside it goes on from there instead of reading everything newer again
(G71). A walk that finds no change leaves the state untouched, so an idle sync writes no new
dlt state.

**Deletion.**

- A root missing from a **complete** walk is emitted as a tombstone (`_deleted: true`) and its
  document is removed from memory on that sync. A history page holds every message between its
  oldest one and the walk position, so a known root in that span that the page lacks is gone;
  it is noted, and forgotten only by the run that completes the walk. A walk that was cut short
  forgets nothing.
- **Memory follows what Slack shows, back to the start of the channel.** A stored root older
  than the oldest message the complete walk got back from Slack is tombstoned too: a person
  deleting the oldest messages, a retention policy and a free plan hiding history past 90 days
  (and deleting it past one year) look the same to the API. The Slack Marketplace guidelines ask
  that an app give no access to a free workspace's messages past those 90 days. A complete walk
  that gets nothing back forgets the whole channel. This is not a time cutoff: the boundary is
  whatever Slack shows on that walk, and a message Slack shows again (a free workspace that
  upgrades) is read and remembered again, at the cost of its extraction.
- A message without replies that is gone marks its day, which is rendered again from what
  Slack shows; a day with nothing left is a tombstone.
- A thread that Slack reports as gone while the walk reads it (`thread_not_found`) is a
  tombstone.
- A deleted reply lowers `reply_count`, so its thread is rendered again without it.
- A root deleted while its replies stay is shown by Slack as a deleted-message placeholder; the
  changed subtype re-renders the thread without it.

**Changes the root cannot show.** An edited reply leaves the root's fingerprint as it was
(Slack bumps none of `reply_count`, `latest_reply` or the root's `edited.ts`), and so do some
reply deletions and a file removed from a message. Through the integration the app's message
events (`message.channels`, `message.groups`) report them: an edited `message_changed`, one
whose files changed, or a `message_deleted` marks the thread in the selected channel
(`sync.mark_thread_changed`, the connection's `changed_threads`, a new stamp per mark), and the
next sync reads that thread again (`slack_source(changed_roots=...)`); the mark is cleared once
the thread was read and that channel's run was stored, so a run cut or failed before that
keeps it, and only if its stamp is still the one the sync read, so a change that arrives while
the sync reads the thread is read by the next one (G72). Marks written before G72 (a list) read
as unstamped; while an older and a newer server run side by side, a mark the older one writes
during a sync can be cleared by it (G88): upgrade with one worker, as Setup asks. Without the app's events
(a standalone `slack_source`), such a change reaches memory only when the thread's fingerprint
next changes. Slack delivers events best effort, so a lost event leaves that change out.

**Known gaps:**

- A name change is picked up only by threads that are rendered again.
- While a walk is unfinished, a channel that gets messages before every run reads its current
  day whole on every run before the walk resumes, so under Tier 1 a busy day leaves the walk
  little of each run and it moves slowly (G71's remainder).
- A channel whose walk takes several runs (a large channel, or Slack's Tier 1 history limits)
  forgets what Slack stopped showing only when a walk completes.
- **A day bigger than one run never completes (G56).** A day's document is built from that
  whole UTC day read in one run, so a cut keeps its position at the end of a day. Under Tier 1
  a run reads about 150 messages (ten pages of 15), so a channel with more than about 140
  messages in one UTC day, the current day included, cannot move past it: its walk never
  completes, nothing older is read and nothing is forgotten. The run reports it under
  `failed_day_too_large` instead of as an ordinary cut, and the integration then waits the
  six-hour interval for that channel instead of resuming after ten minutes. Self-hosted
  internal apps and Marketplace apps are not on Tier 1 (decided 2026-11-10, option 1: report
  it, keep the day as the unit).

**Forget.** `cognee.forget(data_id=...)` removes the document, but the staging row stays, so
the next sync builds it again while the message is still in Slack; `forget(dataset=...)` leaves
the dlt staging copy and state too, so a later sync under the same dataset and resource name
brings everything back as of the last sync. This is SDK-910, for every connector; for Slack it
is accepted (Goran, 2026-11-10): take a message out from Slack (below) or deselect its channel.
A disconnect with `delete_data=true` deletes the dlt staging copy with the dataset (the staged
rows, the sync state and the load packages), so connecting again starts from an empty walk.

**Taking a message out of memory (Goran, 2026-11-10).** *Forget this* or `@cognee forget` on a
message of a synced channel, by its author or by the cognee user who connected the workspace,
records it on the connection (`sync.exclude_message`, `excluded_messages`: `{channel_id: [ts]}`),
marks its thread or its message (as for G6) and starts a sync. Every sync then leaves it out of
the document it would be in (`slack_source(excluded_messages=...)`): its day or its thread is
rendered again without it, and a day or thread with nothing left is tombstoned. It stays in
Slack, and in the dlt staging copy until that sync replaces the row. Anyone else gets an answer
naming who can. A thread's first message taken out leaves its replies in memory.

## Limits

- **A guest in a channel sees its answers.** An answer in a channel uses the selected public
  channels, which every full member of the workspace can read, but a single- or
  multi-channel guest cannot. A guest who asks is told apart (`users.info`'s `is_restricted`
  / `is_ultra_restricted`) and gets an answer from the channel alone, but Slack does not say
  whether a channel has other guests without reading every member, so a guest in the channel
  still reads answers a full member asked for, drawn from public channels the guest is not
  in. Ask in a direct message where that matters, or leave such channels out of the command
  allowlist. A Slack Connect channel is told apart and gets answers from itself alone.
- **Connect from the host the sign-in returns to.** The OAuth callback is accepted only with
  the cookie `authorize` set, and a cookie belongs to one host. Connect (and Link Slack)
  started from a UI that calls the API on another host than `SLACK_REDIRECT_URI` /
  `SLACK_MEMBER_REDIRECT_URI` ends in `error_invalid_state` after Slack's consent page. The
  usual case is a tunnel while the UI is open on `localhost:3000`, which `cognee-cli -ui`
  opens by itself: open the UI on the tunnel's host instead (Setup, step 1). The UI checks
  this before opening Slack and names the address to use.
- **A run is bounded and resumable.** `max_requests` (default 1500) is the request budget of
  one run; a call retried after a rate limit or a 5xx counts once. A spent budget stops the run
  cleanly with `failed_budget`, a rate limit that outlasts the retries with `failed_rate_limit`,
  and, after rows were emitted, an expired token with `failed_auth` and a 5xx or network
  failure that outlasts the retries, or any other error Slack answers with, with
  `failed_unavailable` (before any row they raise; a missing scope always raises). What
  was reached is kept and the next run resumes from the end of the oldest day it reached, which
  it reads again whole. A large channel therefore finishes its first sync over several runs,
  unless one UTC day holds more than a run can read (G56, under Known gaps). Through the
  integration a cut run is resumed ten minutes later.
- **Pause.** A connection paused through its sync policy (`POST
  /api/v1/integrations/slack/sync/policy/pause`, or the owner's default) is not synced by
  the six-hourly tick, the ten-minute resume, a saved selection or a removed message
  (G62). Pausing needs a policy of the connection's own: save one first
  (`PUT /api/v1/integrations/slack/sync/policy`) if it inherits the owner's default.
- **Wall-clock budget.** A run also stops with `failed_budget` once it has run for 600 seconds
  (`MAX_RUN_SECONDS`), checked before each request, so one in-flight request (with its retries)
  can finish past it (a rate-limit wait never runs past the deadline). The walk holds the
  dlt staging lock of the workspace's dataset (other datasets, a CSV upload or another connector,
  stage meanwhile), and this connector waits out short rate limits, so the budget bounds how long
  one run blocks other ingestion into that dataset. The other connectors (Asana, Apollo, Linear) never sleep and
  have no deadline, so there is no sibling value; 600 seconds is the interval after which the
  integration resumes a cut run. The bound per process: one Slack sync runs at a time and each of
  its channels is one run, so an unrelated ingest (another connector, an `add()`) waits for at
  most the channel run in progress, 600 seconds plus that one in-flight request, however many
  workspaces are waiting. A waiting workspace waits for the whole sync ahead of it, up to one
  run per selected channel of that workspace.
- **Rate limits.** A `429` (or `ratelimited`) asking for at most 60 seconds
  (`_MAX_RETRY_AFTER_SECONDS`, the minute Tier 1 asks for) is waited out and retried, up to four
  attempts, unless the wait would run past the run's own 600-second deadline. A longer
  `Retry-After`, or one past the deadline, stops the run at once with `failed_rate_limit`,
  without sleeping, and the next run resumes.
- **Slack's history tier.** For commercially distributed apps that are not on the Slack
  Marketplace, `conversations.history` and `conversations.replies` are Tier 1 (about one
  request a minute, at most 15 messages per call) for apps created on or after 2025-05-29 and
  for every new installation of an older one; installations made before that date keep their
  limits ([changelog](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps/)).
  Marketplace apps keep the usual tiers, and so do internal apps a workspace builds for its own
  use ([clarification](https://docs.slack.dev/changelog/2025/06/03/rate-limits-clarity)). Under
  Tier 1 a large channel takes many runs to walk, and deletions are only forgotten when a walk
  completes. Tier 1's minute is waited out within a run's 600 seconds, so every selected
  channel gets its share of requests on every sync, and one advances as long as each of its
  UTC days fits in a run (G56). While a walk is unfinished, each run first reads what is newer
  than the walk, so a new message does not wait for the walk to end; that read resumes where a
  cut left it, so it never holds the walk back for good (G71).
- **Cost of an unchanged sync.** One `auth.test` plus one history page per 200 channel
  messages (per 15 under Tier 1). A changed thread adds its replies pages, a new author one
  `users.info`, and the first rendered row of a run one `conversations.info`.
- **Missing `users:read`.** Authors are rendered by name with no fallback to raw ids, so a token
  without `users:read` stops the source with `SlackMissingScopeError` ("Reconnect Slack to grant
  users:read"). Through the integration the whole workspace's sync stops before reading
  anything, counted under **`failed_missing_scope`**.
- **Stats** (`cognee_sync_stats`): `scanned`, `emitted`, `skipped`, `deleted`, `failed`,
  `requests`, plus one of `failed_budget`, `failed_rate_limit`, `failed_auth`,
  `failed_unavailable` on a cut run, and `failed_day_too_large` with it when the cut left the
  walk where the previous run left it (G56). The
  integration adds `failed_channel_access`, `failed_channel_unavailable`, `failed_missing_scope`,
  `failed_ingestion` and the non-failures `channels_retired` and `channels_archived`.
- **State size** grows with the channel: about 34 bytes of dlt state per top-level message,
  plus 17 bytes per day that has a day row (G7, computed from the JSON, not measured). Measured on a synthetic channel of 100,000 roots (30% with replies, 10% edited): 3.4 MB
  of state JSON (`state.json` in the dlt working directory) and a 0.94 MB compressed row in
  `_dlt_pipeline_state`. dlt appends such a row for every run that changes the state; an idle
  run changes nothing and appends nothing.
- **What the text leaves out** (decided before the first release, since changing the rendered
  text re-cognifies every synced channel): two people with one full name still merge into one
  entity (G8).
- **Filtering by channel takes its id, not its name (G9).** The node set is
  `slack:channel:<id>`: `recall(node_name=["slack:channel:C0GENERAL"])` filters to one
  channel, `"general"` and `"#general"` find nothing. The name stays out on purpose: node sets
  are part of a row's content hash, so a name in them would re-cognify a whole channel on every
  rename. The ids are in the channel list (`GET /api/v1/integrations/slack/resources`, `id`
  and `name`).
- **A raw search hit names no source (G10).** A chunk returned by `CHUNKS` or `recall` carries
  the stored file name (`text_<md5>`) and the `data_id`, not the title or the link. A search
  with `include_references=True` names each document it read by the title and link its data
  record keeps in `system_metadata` (`title`, `url`), copied into the chunk payload as
  `document_title` / `document_url` (SDK-1500), with the row's channel as `document_parent`, and
  `@cognee` asks that way: each fact ends with a numbered *Sources* line, one link per
  conversation it read, to the message in Slack.
- Direct messages and group DMs are not synced.
- Documents written by the former history importer (`system_metadata.slack_history`) stay in
  the datasets they were imported into and are no longer updated.

## Privacy

This reads the messages of the selected channels, the names of their authors and of the
people mentioned, and the names of attached files (never their content). The text goes to the
configured LLM and embedding providers, and anyone with read access to the target dataset can
see it through search. `@cognee` in a channel answers from what everyone there can read: that
channel when it is synced and the selected public channels, from their messages alone; in a
direct message, from the selected channels the member is in and their own datasets. A
private channel is synced only when both the app and the Slack user who connected
the workspace are members. Errors and logs carry the API method and Slack's error code, never
the token, a channel id, a URL or a response body.

cognee turns off two dlt defaults before it builds a dlt source or pipeline, unless they are
set: `LOAD__DELETE_COMPLETED_JOBS=true`, so a completed load package, which holds the rendered
text of each row, is deleted once loaded (a failed one is kept for the retry), and
`RUNTIME__DLTHUB_TELEMETRY=false`, so dlt sends no telemetry of its own. The staging database
keeps the text of every synced row until the channel is retired or the connection is
disconnected with `delete_data=true`. A deleted conversation disappears from the graph and,
once its tombstone is loaded, from staging. An answer that quoted a message stays in the
search history, and as a message in Slack, after the message is deleted in Slack.

## Testing

```bash
pytest cognee/tests/unit/tasks/connectors/test_slack_source.py \
       cognee/tests/unit/tasks/connectors/test_slack_client.py \
       cognee/tests/unit/modules/integrations/test_slack_sync.py \
       cognee/tests/unit/modules/integrations/test_slack_integration_routes.py
pytest cognee/tests/integration/tasks/connectors/test_slack_ingestion.py \
       cognee/tests/integration/tasks/connectors/test_slack_forget.py
```

The unit tests drive the walker with a plain dict and the dlt behaviour through a real
`dlt.pipeline` on SQLite, against an in-memory Slack (`fake_slack.py`); the client tests go
through `httpx` with a mocked transport. The integration tests run real `add` and `cognify`
with a mocked LLM. None of them needs a Slack workspace.
