# Telegram SDK connector

A Telegram data-source connector for [cognee](https://github.com/topoteretes/cognee): sync the
messages a bot sees in the groups, supergroups and channels it is in into memory,
incrementally, and forget a chat when the bot is removed from it or the chat leaves your
selection.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each message becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. One source is one bot token.

Ported from the cognee-community connector (PR #355, commit `76ff1a4`).

## Requirements

Bundled with the Cognee SDK (a core that does not read per-row structure,
`DOCUMENT_SYNC_VERSION` below 3, is refused). No extra and no separate package: the Bot API
client is the standard library's `urllib`.

**Auth is a bot token from @BotFather.** Telegram has no OAuth for bots, so there is no
authorize step and no access token to refresh: a host stores the token the user pasted and hands
it to `_PROVIDERS["telegram"]` (`cognee.modules.integrations.ingestion`), whose service builder
wraps it. A token does not expire; a revoked one surfaces as `failed_auth`. Key the connection
by `telegram_account_id(service, check_active=...)`, the bot's id from `getMe`: it is the id the
source checks its state against, so connecting another bot gives it its own table instead of one
whose state belongs to the first bot (refused with `failed_bot_mismatch` on every run).

Use a **dedicated bot** for cognee:

- Its updates are read with `getUpdates`, which fails with HTTP 409 while the bot has a webhook
  or while another process polls it. The connector never deletes a webhook.
- In groups a bot only sees what privacy mode allows: commands and replies to it. Disable
  privacy mode in @BotFather (`/setprivacy`) or make the bot an admin to sync every message.
  The connector warns when privacy mode is on.
- A bot only sees messages sent after it joined. There is no history backfill.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import telegram_source

await cognee.remember(
    telegram_source(token="123456789:AA...", chats=[-1001234567890, "@my_channel"]),
    dataset_name="telegram",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
    self_improvement=False,
)

answer = await cognee.recall("What did Ana say about the release date?", datasets=["telegram"])
```

Run the same `remember` call again whenever you want to sync, at least hourly for a busy bot
(see Limits).

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: the staging table, the dlt state and the pipeline scope. Lowercase letters, digits and underscores, starting with a letter | `telegram_<bot id>` with `token`; required with `service` |
| `chats` | Chats to sync: a non-empty list or tuple of numeric chat ids (`-1001234567890`, or the same as a string) and `@username`s. `None` syncs every group, supergroup and channel the bot is in | `None` |
| `token` | The bot token, `<bot id>:<secret>`. Never read from the environment | needed unless `service` is given |
| `service` | Pre-built client, from `build_telegram_service(bot_token)`. A client of your own must accept `call(method, params, *, check_active, budget)` and run `check_active` before each attempt it makes | built from `token` |
| `check_active` | Host checkpoint, called before every request attempt and every retry wait, and around every row. What it raises stops the run as it is | none |
| `include_private` | With `chats=None`, also sync private chats (messages people send the bot) | `False` |

`chats` is sorted and repeats are dropped. A bare string (`"@my_channel"`), an empty list, a
bool, a blank entry or anything that is not a chat id or a username raises `ValueError` when the
source is built. A token of the wrong shape raises `ValueError` without echoing it, so a pasted
fragment of a secret never becomes a table or state name. Pass `token` or `service`, not both.
Building the source makes no request.

**Private chats are off by default.** They are personal conversations with whoever writes to the
bot, so `chats=None` skips them unless `include_private=True`. A private chat whose id is listed
in `chats` is synced either way. A private chat is selected by id only: the user's handle is
never kept in state.

**A `@username` names a chat, not a name.** The first time a selected username is seen on a
group, supergroup or channel, it is bound to that chat (state key `username_chats`), and from
then on it selects that chat and its upgrade chain whatever the chat is called: a chat that
changes its username keeps syncing, and another chat that later takes the freed username is not
selected by it (its messages count as `skipped`). Replacing the username with the chat's id in
`chats` forgets nothing; dropping it forgets the chat. A username
with no binding yet is looked up at the start of the run with `getChat` (`chat_id="@name"`), so
it binds to its chat at once, before any message of that chat arrives. A username is bound from the saved chat meta only
when exactly one chat there carries it, and the chat last seen claiming a username takes it from
every other chat's meta, so a chat that released the name is never pulled in by it.

**Swapping a username keeps the chat.** Replacing `@old` with the same chat's `@new` forgets
nothing: `getChat` binds `@new` to the chat on the same run, before the selection is compared.
Telegram may answer HTTP 400 "chat not found" for `@new` right after a rename, so that answer
alone deletes nothing: every stored group or channel the run would drop is then asked for by its
id (`getChat` with the chat id), whatever the old selection named: ids, usernames or every chat,
and when it already carries `@new` it is bound to it and keeps its history. Only when the chat
answers with another username, or not at all (any 400, or 403: the bot is no longer in it), is
the name taken for a typo: it selects nothing and the chat it replaced is forgotten on that run.
A name that is not found is asked again on every run until it binds, one request a run (plus one
per group or channel the run would drop, on the run that changes the selection: narrowing from
every chat to one name the lookup cannot see yet costs one request per stored group and channel,
and a rate limit repeats them on the next run, forgetting nothing meanwhile; a private chat is
never asked, since no answer could keep it). This relies on getChat by id already showing the
new username: when Telegram still answers with the old one, the chat is forgotten (a known
limit; put the chat's id in `chats` next to the new name to keep it). A bound username is never
looked up again, so a name another chat takes over later still names the first chat. Write
usernames as `@name`; a `t.me/` link is refused. A username that names a person selects nothing
(a private chat is selected by id only). Any other `getChat` error stops or raises the run like
the other requests, and nothing changes.

## What is ingested

One document per message that has text or a caption.

| Column | Value |
|---|---|
| `id` | `<chat id>:<message id>` (message ids are only unique inside a chat) |
| `title` | `<first line> (<chat> #<message id>)`, e.g. `ok (Team chat #2)`; the first line is shortened to 80 characters, the full text stays in `content` |
| `content` | the header lines, a separator line, then the text |
| `url` | `https://t.me/<username>/<id>` for a public chat, `https://t.me/c/<id>/<id>` (members only) for a private supergroup or channel, none for a basic group or a private chat |
| `cognee_node_set` | `["telegram:chat:<root chat id>"]` |
| `cognee_structure` | `{"ancestors": [{"kind": "chat", "id": "<root chat id>", "name": "<chat title>"}]}` |

```
Chat: Team chat
From: Ana Kovac
Handle: @ana_k
Date: 2026-10-09 08:53 UTC
Forwarded from: Bob
Edited: 2026-10-09 09:10 UTC
In reply to Bob: Where is the doc?
--- message ---
Here it is.
```

`Handle`, `Forwarded from`, `Edited` and `In reply to` appear only when they apply. Every name,
title and quoted reply is collapsed to one line, and the text comes after the fixed
`--- message ---` line, so a member cannot write a line that reads as another header. The
`From:` line holds the display name alone (first and last name, as the person set them) and the
handle sits on its own line, so the extractor sees the same name string another connector renders
for the person (Linear's `Assignee: Ana Kovac`, Slack's full name); with an extractor that reads
a name off its line, the Telegram author and the Linear assignee are one entity. A person who set
only a first name is `From: Ana` and will not match a full name elsewhere, and an extractor that
also picks up `@ana_k` makes the handle an entity of its own. The layout is frozen once released.
A channel post's author is the channel.
A forwarded message is credited to its original author on `Forwarded from:`, while `From:` stays
the forwarder. A quoted reply is cut at 200 characters (display only: the quoted message is its
own document). No numeric user id and no phone number is rendered; a sender without a name or
handle is `unknown`. A media message is its caption, prefixed with the kind (`[photo] ...`);
files are never downloaded. Service messages (joins, pins) and media without a caption are no
document.

The rendering is deterministic, so an unchanged message keeps its content hash. Telegram never
sends an unchanged message again anyway: every update arrives once.

**Node set and chat container.** Every message row carries `cognee_node_set` so a `recall` can
be scoped to one chat (`node_name=["telegram:chat:-1001234567890"]`), and `cognee_structure`,
which makes the chat a container with a `child_of` edge from each message. Both name the chat by
the **root of its upgrade chain**: when a group becomes a supergroup it gets a new id, and both
ids are one chat. The upgrade announcements of a page are read before its messages, so a
supergroup message that comes ahead of the announcement in the same page still counts as the old
chat. Telegram sends the upgrade before the supergroup's first message, so an announcement that
only arrives in a later page needs updates out of order; if it happens, the earlier messages keep
the new id as their node set, and with `chats` naming only the old id they are skipped. Only ids
are used, so renaming a chat changes no node set; the container's name follows the title and is
not part of a document's identity. A tombstone carries neither.

**Titles are unique per message.** The first line says what a message is about, and the chat and
message id after it tell two messages that both say "ok" apart, so each is its own source in a
search's references, also in a basic group, which has no message links. The chat is named as on
the `Chat:` line (title, else `@username`, else id), except a private chat, which both the title
and the `Chat:` line name by the person's name, never their handle. An edit keeps the message id, so its title
changes only with its first line.

### Frozen names

These are permanent once anyone has synced, because every change re-extracts every message on
your LLM key or orphans documents: the provider and table prefix `telegram` (hosted names are
`telegram_<24 hex>`, the default `telegram_<bot id>`), the row id `<chat id>:<message id>`, the
node set `telegram:chat:<root chat id>`, the title format, the rendered text above, the state
keys and the stats keys below. Telegram never sends a message again, so a row keeps the title
and text it was rendered with until the message is edited.

## How sync and forget-on-delete work

**One page per run.** A run calls `getMe` (to check the token and the owner of the state),
`getChat` once for each selected `@username` not bound yet, and then `getUpdates` once, for up to 100 updates (the Bot API maximum). Asking for the next page
confirms the previous one on Telegram's side, and confirmed updates are gone for good, so a run
never asks for a second page: its updates are confirmed by the *next* run, whose offset comes
from the state dlt saved with this run's rows. A host revoking access between rows, or a run
that stops or raises, saves no state, so the next run reads the same page. A load that fails
after the run emitted its rows leaves dlt's package with those rows and their cursor in place,
and the next sync of the same pipeline loads that package first (it reads no new page in that
run). Either way nothing is lost.

**After a quiet week** Telegram may restart `update_id` at a random value below the stored
cursor, so after six days without updates the connector reads without an offset, which confirms
nothing. The six days run from the newest update's own `date` (an edit's `edit_date`), Telegram's
clock, not from when the run read it.

**Edits** arrive as `edited_message` / `edited_channel_post` with the same message id and
replace the row; the new text is extracted, the old one leaves memory. An edit that removes the
text (a caption deleted from a photo) tombstones the stored message.

**Forget-on-delete is per chat.** Only three things forget data: the bot being removed from a
chat, an edit that takes the text away, and a chat leaving your selection.

**Deleted messages and deleted chats stay in memory.** The Bot API sends a bot no event when a
message is deleted in an ordinary chat (checked on a live bot by the community port: nothing
arrives, not even a gap in `update_id`), and none is known for a chat that is deleted. Such a
message or chat stays in memory until you remove it yourself: take the chat out of `chats`,
remove the bot from the chat, or `forget` the data. This is an accepted limit of the Bot API,
not something a later sync repairs.

| Event | Effect |
|---|---|
| The bot is removed from a group, supergroup or channel (`my_chat_member` `left` / `kicked`, or `restricted` with `is_member` false) | every message stored for the chat and the chats it was upgraded from is tombstoned |
| A chat leaves the selection (`chats` narrowed, or `include_private` turned off) | the same, once, on the next run |
| An edit removes the text | that message is tombstoned |
| A user blocks the bot in a private chat (`kicked` in a private chat) | nothing: blocking a bot to stop notifications is not a request to forget |
| A message is deleted in Telegram | nothing: the bot is not told |
| A chat is deleted in Telegram | nothing, unless the bot's removal from it arrives as `my_chat_member` |

A tombstone names the stored messages by reading the ids of the connection's own staging table
(the table named after `resource_name`, in the pipeline that runs the source), plus the rows
emitted earlier in the same page, which that load has not written yet. The staging table is the
one record of what is stored, since a tombstone removes its row there. A run reads it at most
once, and only when it has to name stored messages: the bot removed from a chat, a chat that
left the selection, an edit that took the text away. A run of new messages never reads it. Before
the first load the table does not exist, which counts as nothing stored. Reading it needs a dlt
pipeline, so a removal in a source iterated outside one (`list(resource)`) raises
`TelegramStagingError`; a read that fails (any dlt database error other than the missing table)
raises the same error with `failed` and `failed_provider` set, and the next sync retries the same page. The same holds
on Postgres staging, where before the first load the schema is missing too. On SQLite, dlt keeps
the staging files in WAL mode, so another process writing to them does not block the read; one
that holds them exclusively does, and the run fails with `TelegramStagingError` (its message names
the exception class only) and keeps its cursor. A load that another writer blocks fails in dlt's
load step after the source has emitted its rows; dlt keeps the package and loads it first on the
next run of the same pipeline. A chat put back into the selection brings only new messages:
Telegram never sends the old ones again.

**State keys** (`state["v"] = 1`): `last_update_id`, `last_update_seen_at`, `bot_id` (the bot
that owns the state; a state of another bot raises), `migrated_from` (`{new chat id: old chat
id}`), `chats_meta` (`{chat id: {type, username}}`, the username only for groups and channels, so
a `@username` entry can be bound to a chat it has not been seen on in this run), `selection` (the
last run's selection), `username_chats` (`{username: [chat id]}`, the chat each selected
`@username` is bound to; a state written before this key existed binds from `chats_meta` on its
next run, so no rename is missed). `chats_meta` keeps only the selected chats, the ones a later
run reads. No message id is kept in state: dlt keeps every state version, so a list that grew
with the corpus cost space quadratic in the number of messages (60 runs of 100 messages made
571 KB of state rows). The state grows with the number of chats, not messages.

### Stats

`resource.cognee_sync_stats` holds counts only, every key present from the start and on every
exit:

| Key | Meaning |
|---|---|
| `scanned` | message rows emitted |
| `deleted` | tombstones emitted |
| `skipped` | updates that became no row: a chat outside the selection, a message without text, an update type the connector does not read |
| `backlog` | 1 when the run read a full page: more updates may be waiting |
| `gap` | update ids that may have been lost: the ids skipped between the saved cursor and this run's first update, counted only when the last update the connection saw is more than 24 hours old. 0 otherwise |
| `failed` | 1 when the run stopped or raised, with one reason key below |
| `failed_auth` | Telegram rejected the token (HTTP 401 or 404). Run stops, nothing changes. A bot token never expires: it was revoked in @BotFather, or the bot was deleted |
| `failed_conflict` | `getUpdates` answered 409: a webhook is set, or another process polls this bot. Run stops, nothing changes |
| `failed_rate_limit` | a 429 the retries did not get past, or whose wait does not fit the run's wait budget. Run stops, nothing changes |
| `failed_unavailable` | a 5xx or no answer after the retries. Run stops, nothing changes |
| `failed_provider` | any other failure Telegram's side caused: another 4xx, an HTTP 200 that is not a Bot API answer, a `getMe` without a numeric bot id, a staging read that fails, a state of a newer version. The run raises |
| `failed_unexpected` | a failure that is not Telegram's (a defect of this source), at any point of the run, raised as `TelegramUnexpectedError` with the original as its cause. The run raises and the cursor stays |
| `failed_bot_mismatch` | the state of this `resource_name` belongs to another bot (`TelegramBotMismatchError`). The run raises |

A run that stops keeps its cursor: nothing was confirmed and the next sync reads the same page.
A host that counts a raised run itself reads the key with `classify_sync_error(error)` (the
`telegram` entry of `_PROVIDERS` carries it), so one failure is counted once.

**`gap` is a possible loss, not proof.** Telegram keeps an update nobody collected for about 24
hours, and `update_id` counts up by one per update, so a hole between the saved cursor and the
next update can mean updates expired unread. It can also be updates Telegram never sends this
bot: the connector asks only for messages, posts, their edits and its own membership, and
Telegram skips the ids of every other update type (a reaction, a poll answer, a callback query).
A hole alone therefore proves nothing; only a hole after more than 24 hours since the last update
the connection saw is counted, since nothing newer can have expired. It is not counted on the
first run, after six quiet days when the first id is at or below the cursor (ids restart only
lower, so a higher first id is counted), or between the updates of one page, and it changes nothing about what the run emits. When it shows up, sync more often. It also
counts after a day in which the bot's chats were quiet but other update types arrived, so a
nonzero value after a quiet day may be nothing.

## Limits and their sources

| Bound | Value | Source |
|---|---|---|
| Updates per run | 100 | the Bot API's `getUpdates` `limit` maximum; there is no knob to read less |
| Requests per run | 2 (`getMe`, `getUpdates`), plus one `getChat` per selected `@username` not bound yet, and, on a run where a selected `@username` is not found, one by id per stored group or channel the run would drop | this connector: a second `getUpdates` would confirm a page before it is loaded |
| Unconfirmed updates kept | about 24 hours | Telegram: updates "will not be kept longer than 24 hours" |
| Wait on a 429 | Telegram's `retry_after` | Telegram |
| Attempts per request | 5 | our decision; it only delays a run |
| Backoff on a 5xx or no answer | 1, 2, 4, 8 s | our decision; it only delays a run |
| Sleep per run, all requests together | 60 s | our decision (OneNote waits at most 60 s a run too): the walk holds the process-wide staging lock while it sleeps. A wait that does not fit stops the run with `failed_rate_limit` (`failed_unavailable` for a 5xx); it delays the sync, it drops nothing |
| Socket timeout | 30 s a request | our decision; Telegram documents no latency |
| Title, quoted reply | 80, 200 characters | display only; the full text stays in the document |

**Throughput.** One page per run means at most `100 x runs per day` updates per bot, and
Telegram drops undelivered updates after about 24 hours. Sync at least hourly for chats with
more than 100 messages a day (2,400 updates a day per bot); a daily sync keeps up with 100. A
run that read a full page reports `backlog=1` and logs a warning; when it repeats, sync more
often, or updates will be lost for good. A sync that comes more than 24 hours after the last
update it saw reports the skipped update ids it finds in `gap` (see Stats).

**Several bots** in one chat see the same keys, so each bot needs its own `resource_name` (the
default `telegram_<bot id>` already is). Two connections polling the same bot steal each other's
updates: each `getUpdates` offset confirms what the other has not read. Do not connect one bot
twice. Two different bots in one chat synced into one dataset give two documents per message,
one per bot's table.

## Forget and its gaps

- `forget(data_id=...)` removes one message from memory, but its staging row stays and the next
  sync builds it again. Remove the chat from the selection, or the bot from the chat, instead.
- `forget(dataset=...)` removes memory; the staging copy and the dlt state stay, and the next
  sync rebuilds every message as of the last sync. For Telegram the staging table is the only
  copy of the corpus (nothing is ever delivered twice), so that is also how memory is recovered.
  A full erase needs the staging table and dlt's working directory removed too.
- `forget(memory_only=True)` then `cognify()` rebuilds the graph from the stored documents.

## Privacy

This connector reads every message the bot can see in the selected chats: message text,
captions, the display names and `@handles` of senders, forwarded authors and the people replied
to, chat titles, and with `include_private=True` private chats. All of it goes to your
configured LLM and embedding providers, and anyone with read access to the dataset can see it.
Group members never consented to that individually: tell them the bot is there. Logs carry
counts only; no chat title, chat id, name or text is logged at info or above.

## Testing

Unit tests drive the source with an in-memory Bot API (`fake_telegram.py`) and through a real
`dlt` pipeline on SQLite, and the client through a patched opener and a real local HTTP server.
Integration tests run the real `cognee.add()` and `cognify()` with the LLM and embeddings
mocked. CI runs them on SQLite; they also pass with Postgres as the relational and staging store:

```bash
pytest cognee/tests/unit/tasks/connectors/test_telegram_source.py \
       cognee/tests/unit/tasks/connectors/test_telegram_client.py \
       cognee/tests/integration/tasks/connectors/test_telegram_ingestion.py \
       cognee/tests/integration/tasks/connectors/test_telegram_forget.py
```

This port has not been run against a live bot; the community connector was (PR #355).
