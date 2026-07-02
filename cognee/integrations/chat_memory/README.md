# Chat Memory Adapter

A small, framework-agnostic layer that every cognee-powered chat bot (Slack,
Telegram, Discord, a personal "second brain", and so on) plugs into. Each bot
stays thin, around 100 lines, and they all share one consistent memory model
built on cognee's `remember` / `recall` / `forget` primitives.

```
platform event  ->  Conversation + Message  ->  ChatMemoryAdapter  ->  cognee
                                                (ingest / answer / forget)
```

## Conversation scoping

A conversation maps to a [`Scope`](./models.py) that holds two keys, matching
cognee's two storage knobs:

| key | what it is | used for |
| --- | --- | --- |
| `dataset` | durable graph + recall boundary; the unit `forget` wipes | what is shared |
| `session` | fast per-conversation cache | what is recent |

The adapter keeps these as separate fields because one value cannot serve every
bot:

- **Per-channel team bot:** `dataset = chat:{platform}:{workspace}:{channel}`.
  Everyone's messages land in one connected graph, so "what did we decide about
  X" can traverse a decision across who proposed it, who agreed, and what
  superseded it.
- **Per-user second brain:** `dataset = brain:{user}`, decoupled from the
  channel, so a note captured in Telegram is recallable from the web. The
  `session` still follows the transport for recent context.

Pick a strategy from [`scoping.py`](./scoping.py) (`per_channel_scope`,
`per_user_scope`, `per_workspace_scope`) or pass your own `Conversation -> Scope`.

## Build your own bot in 5 minutes

**1. Install** (the console demo needs no keys; a real bot needs an LLM key):

```bash
pip install cognee
```

**2. Create an adapter.** For zero-setup local dev use the in-memory backend;
for real graph memory use `CogneeMemoryBackend` and set `LLM_API_KEY`.

```python
from cognee.integrations.chat_memory import (
    ChatMemoryAdapter, Conversation, Message,
    per_channel_scope, InMemoryMemoryBackend,   # swap for CogneeMemoryBackend()
)

memory = ChatMemoryAdapter(scope=per_channel_scope, backend=InMemoryMemoryBackend())
```

**3. Map your platform event to a `Conversation`.** This is the only
platform-specific code you write:

```python
def conversation_of(event) -> Conversation:
    return Conversation(
        platform="myapp",
        workspace=event.team_id,
        channel=event.channel_id,
        user=event.user_id,
        thread=event.thread_id,
    )
```

**4. Wire the three primitives:**

```python
# remember every message (background, returns immediately)
await memory.ingest(conversation_of(evt), Message(text=evt.text, user=evt.user_id,
                                                  timestamp=evt.ts, permalink=evt.link))

# answer a question, with citations back to the source messages
answer = await memory.answer(conversation_of(evt), "when do we ship?")
reply(answer.text, sources=[c.permalink for c in answer.citations])

# privacy: forget me / forget this scope
await memory.forget(conversation=conversation_of(evt), user=evt.user_id)  # forget me
await memory.forget(conversation=conversation_of(evt))                    # wipe the scope
```

That's the whole bot. See [`examples/integrations/chat_memory/`](../../../examples/integrations/chat_memory/)
for a runnable console bot (no keys) and a ~100-line Telegram bot.

## Privacy: consent and "forget me"

- **Opt-in / opt-out.** `ingest` is gated on consent. In a group/channel the
  default is *deny until opt in* (one member can't consent for everyone); in a
  1:1 DM the default is *allow* (using the bot is the opt-in). Override per user
  with `adapter.set_consent(user, on)`.
- **Forget me.** `forget(conversation=..., user=...)` deletes just that user's
  items (resolved via the `user` stamped into each item's `external_metadata`)
  and revokes their consent. `forget(conversation=...)` wipes the whole scope.

> **Note on shared graphs.** In a shared dataset, cognify can merge facts from
> several users into one graph node. Per-user forget removes the user's own
> items; fully *dedup-aware* deletion (drop a node/edge only when no other
> user's data references it) is the planned follow-up and belongs in this core.
> Per-user datasets (the second brain) are unaffected: their forget-me is a
> whole-dataset wipe with nothing shared to orphan.

## Transports

`CogneeMemoryBackend` is the in-process Python-SDK reference implementation (the
fast path, no network hop). The adapter talks to it through the four-method
[`MemoryBackend`](./backend.py) interface, so a TS-client or MCP-backed backend
can satisfy the same contract later without touching the adapter or any bot.
`InMemoryMemoryBackend` implements the same interface with plain dictionaries
for tests and offline development.

## Testing

The adapter is exercised end-to-end against `InMemoryMemoryBackend`, so the
suite is fully deterministic and needs no LLM, database, or API keys:

```bash
pytest cognee/tests/unit/integrations/chat_memory/
```
