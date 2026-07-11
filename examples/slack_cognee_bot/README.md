# Slack bot powered by cognee memory

A [Slack Bolt](https://slack.dev/bolt-python/) app that quietly turns a channel's
conversation into [cognee](https://www.cognee.ai/) memory, so your team can ask
**"@cognee what did we decide about X?"** and get an answer **with links back to
the source messages**.

> Example app for [issue #3609](https://github.com/topoteretes/cognee/issues/3609).
> It lives in `examples/` and is not part of the cognee core install.

## What it does

- **Silently ingests** messages from channels you opt in — each message becomes a
  cognee data item, tagged to that channel.
- **Answers questions** from the channel's memory via an **@mention** or the
  **`/recall`** slash command, replying with a Block Kit message that lists the
  **source messages as clickable permalinks** (citations).
- **Forgets on request** — **`/cognee-forget`** deletes the whole channel's memory.
- **Opt-in / opt-out** per channel with **`/cognee-optin`** and **`/cognee-optout`**;
  the bot only ingests channels you've opted in.

## Prerequisites

- A working cognee install and an LLM key (`LLM_API_KEY`) — see the repo's main
  README / `.env.template`. cognee's local defaults (SQLite + LanceDB + Ladybug)
  are enough; no extra services required.
- Python 3.10–3.14.
- The **`slack`** extra (adds `slack-bolt`):

  ```bash
  uv pip install -e ".[slack]"
  ```

- A Slack app in **Socket Mode** (no public URL needed):
  1. Create an app at <https://api.slack.com/apps> → **From scratch**.
  2. **Socket Mode** → enable it → create an **App-Level Token** with the
     `connections:write` scope (this is your `SLACK_APP_TOKEN`, `xapp-...`).
  3. **OAuth & Permissions** → add **Bot Token Scopes**:
     `app_mentions:read`, `channels:history`, `chat:write`, `commands`.
     Install the app to your workspace → copy the **Bot User OAuth Token**
     (`SLACK_BOT_TOKEN`, `xoxb-...`).
  4. **Event Subscriptions** → subscribe to bot events: `app_mention`,
     `message.channels`.
  5. **Slash Commands** → create: `/recall`, `/cognee-optin`, `/cognee-optout`,
     `/cognee-forget`.

## Run it (5 minutes)

```bash
# 1. Install cognee with the slack extra
uv pip install -e ".[slack]"

# 2. Configure env: copy the template and fill in your keys/tokens
cp .env.template .env
# set at least: LLM_API_KEY, SLACK_BOT_TOKEN, SLACK_APP_TOKEN
# (see the "Slack bot example" section at the bottom of .env.template)

# 3. Load the env vars into your shell
set -a && source .env && set +a
# (or run with: uv run --env-file .env python -m src)

# 4. Start the bot
cd examples/slack_cognee_bot
python -m src
```

Then, in Slack:

1. **Invite** the bot to a channel (`/invite @your-bot`).
2. Run **`/cognee-optin`** — the bot posts a one-time disclosure and starts
   remembering new messages.
3. Chat normally for a bit.
4. Ask **`@your-bot what did we decide about the launch?`** or
   **`/recall who owns billing?`** — you'll get an answer with source links.
5. Done? **`/cognee-optout`** stops ingestion; **`/cognee-forget`** deletes the
   channel's memory entirely.

## Configuration

All settings are environment variables (see `.env.template`). Only the tokens are
required; the rest have sensible defaults.

| Variable | Purpose | Default |
| --- | --- | --- |
| `SLACK_BOT_TOKEN` | Bot token (`xoxb-…`) for Web API calls | — (required) |
| `SLACK_APP_TOKEN` | App-level token (`xapp-…`) for Socket Mode | — (required) |
| `COGNEE_SLACK_DEFAULT_TEAM_ID` | Workspace id fallback for the session id | `""` |
| `COGNEE_SLACK_OPTED_IN_CHANNELS` | Comma-separated channels to ingest at startup | `""` |
| `COGNEE_SLACK_COGNIFY_BATCH` | Messages buffered before a cognify runs | `10` |

## How it works

Each channel maps to its own cognee dataset (`slack_<channel_id>`). Ingested
messages are buffered and turned into a knowledge graph in batches
(`cognify`). Answers run a graph-completion search for the prose plus a chunk
search for the sources; each source chunk carries a `document_id` we set at
ingest time, which we join back to the original message's permalink via a small
local index — that's how citations are produced.

The bot talks to cognee through a thin local **chat-memory adapter**
(`ingest` / `flush` / `answer` / `forget`). This implements the shared adapter
pattern from [issue #3608](https://github.com/topoteretes/cognee/issues/3608)
locally; when that shared core lands, only the adapter implementation is swapped
— the Slack layer is unchanged.

## Limitations (read me)

- **Answers reflect the last flush, not every message instantly.** Ingestion is
  batched — a channel is cognified when it hits `COGNEE_SLACK_COGNIFY_BATCH`
  messages or right before a question is answered. This keeps cost sane (a full
  `cognify` per message would be prohibitively expensive).
- **Forget is channel-level only.** `/cognee-forget` deletes the whole channel
  dataset. cognee has no per-user delete today, so a per-user "forget me" is
  **not** supported — it's a follow-up.
- **Scope is per channel.** Memory and answers are isolated to the channel you're
  in; the bot doesn't reason across channels.
- **Citations aren't persisted across restarts.** The `document_id` → permalink
  map is kept in memory. After a restart the cognee graph (and answers) survive,
  but a source message whose permalink is no longer known degrades to a
  plain-text citation (never a broken link) until it is ingested again.
- **#3608 is local.** The chat-memory adapter is implemented in this example
  until the shared #3608 core is merged.

## Development

Run the example's test suite (no Slack tokens, no cognee keys, no network — the
Slack SDK and cognee calls are mocked):

```bash
uv run pytest examples/slack_cognee_bot/tests/ -v
```
