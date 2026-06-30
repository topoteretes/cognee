# cognee-telegram

A Telegram bot where **each chat is a memory**. Forward it articles, drop notes,
chat normally — then `/ask` and get answers **with citations back to the original
messages**. No OAuth, no public URL: it runs from your laptop with long polling.

Built on cognee's `remember` / `recall` / `forget` ([#3610](https://github.com/topoteretes/cognee/issues/3610)).

---

## Run your own in 5 minutes

**Prerequisites**
- Python 3.10–3.14
- A Telegram bot token — message [@BotFather](https://t.me/BotFather), send `/newbot`, copy the token (free, ~1 min, no phone number shared with anyone).
- An LLM key for cognee to build/query memory — an OpenAI key works out of the box. (Costs a few cents while you test; cognee defaults to `openai/gpt-5-mini`.)

**1. Install**
```bash
cd cognee-telegram
uv venv && source .venv/bin/activate
uv pip install -e .
```

**2. Configure** — copy the template and fill in two values:
```bash
cp .env.template .env
# edit .env:
#   TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
#   LLM_API_KEY="sk-..."
```

**3. Run**
```bash
set -a && source .env && set +a   # export the .env vars
python -m cognee_telegram
```

**4. Use it** — open Telegram, find your bot, and:
```
You:  The Q3 review is on Friday at 2pm in room 4.
You:  /ask when is the Q3 review?
Bot:  The Q3 review is on Friday at 2pm in room 4.
      Sources:
      • "The Q3 review is on Friday at 2pm in room 4."
      (from graph memory)
```

## Commands

| Command | What it does |
|---|---|
| *(any message)* | Captured into this chat's memory |
| `/ask <question>` | Answer from this chat's memory, with sources |
| `/forget` | Wipe this chat's memory (graph + vectors) |
| `/optout` / `/optin` | Pause / resume capturing in this chat |
| `/start` · `/help` | Show the intro |

## How it works

- **One chat → one cognee dataset** — a DM is `telegram_dm_<user_id>`, a group is
  `telegram_group_<chat_id>` (forum topics add the thread). The dataset is the memory
  boundary `/forget` clears.
- **Durable graph memory** — messages are ingested with `remember(dataset_name=...)`, which
  runs `add` + `cognify` to build a queryable knowledge graph for the chat. `/ask` then runs
  `recall` over that graph. (Buffer with `COGNEE_TG_BATCH_SIZE` so a busy group triggers one
  graph build per batch instead of per message.)
- **Citations** — the bot keeps a `message → (chat_id, message_id)` ledger; when `recall`
  grounds an answer, the bot maps it back to the originating message and renders a
  `t.me/c/...` deep link (supergroups) or quotes the snippet (DMs / basic groups).
- **`/forget`** — `forget(dataset=...)` clears the chat's graph + vectors completely, and the
  bot drops its local ledger for that chat. Asking before any capture (or after `/forget`)
  returns "nothing here yet" rather than erroring.

The memory logic lives in `CogneeMemoryAdapter` (transport-agnostic), so the same core can
back a Slack/Discord bot later — the Telegram layer is just I/O.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** @BotFather token. |
| `LLM_API_KEY` | — | **Required** (by cognee) to build/query memory. |
| `COGNEE_TG_PER_USER` | `false` | Split group memory per sender (hard per-user delete). |
| `COGNEE_TG_BATCH_SIZE` | `1` | Buffer N messages before one cognee write (raise it for busy groups). |
| `COGNEE_TG_INGEST_DEFAULT` | `true` | Capture by default until a chat runs `/optout`. |

## Tests

Deterministic, **no real keys** — Telegram is mocked with `unittest.mock` and cognee's
memory API is patched at the boundary:
```bash
uv pip install -e ".[dev]"   # or: uv pip install pytest pytest-asyncio
pytest -q
```

## Scope

**v1 (this):** passive capture, `/ask` with citations, `/forget`, `/optout`, tests, this guide.
**Later:** link/media extraction, `/forgetme` + per-user scoping, message edit/delete sync,
and consuming structured citations + abstention from [#3604](https://github.com/topoteretes/cognee/issues/3604) as they land.
