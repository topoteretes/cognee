# Web chat widget powered by cognee memory

An embeddable chat widget — one `<script>` tag — that answers questions from
your docs/site with **cited** sources, remembers each visitor's conversation
in isolation, and lets anyone **forget** their chat or opt out entirely.

It's built on a thin, framework-agnostic [`ChatMemoryAdapter`](adapter.py)
(`ingest` / `answer` / `forget`) so the same core can back a WhatsApp or MS
Teams bot later without touching the transport. This is the web transport for
issue [#3612](https://github.com/topoteretes/cognee/issues/3612), and is
shaped to fold onto the shared adapter core ([#3608]) when that lands.

## How memory is scoped

Each browser conversation maps to a stable cognee `session_id`:

```
web:{site_id}:{visitor_id}:{conversation_id}
```

- **Default boundary: per visitor-conversation.** Turns are stored in cognee's
  *session cache* (scoped by `session_id`), so one visitor's chat never leaks
  into another's.
- **"Ask our docs" mode:** your docs are seeded once into a shared, read-only
  dataset `web:{site_id}:docs`. Every conversation can `recall` from it; none
  can write to it. Clean split between shared knowledge and personal memory.
- **Citations:** answers come back via `recall(..., include_references=True)`
  and the widget renders each source inline.
- **Forget / opt-out:** the `/forget` command (and the widget's *Forget me*
  link) clears just that conversation; unticking *Remember this chat* stops
  any ingestion.

## Run your own in 5 minutes

```bash
# 1. Install cognee (from the repo root)
uv pip install -e .

# 2. Point cognee at an LLM (OpenAI shown; any supported provider works)
cp .env.template .env
# edit .env -> LLM_API_KEY="sk-..."

# 3. Start the widget backend (seeds a small demo docs corpus on boot)
uv run python examples/bots/web_widget/server.py

# 4. Open the "ask our docs" demo
open http://localhost:8000
```

Ask *“What is cognee?”*, then type `/forget` to wipe the conversation.

### Embed on your own site

```html
<script src="http://localhost:8000/widget.js"
        data-site-id="acme"
        data-api="http://localhost:8000"></script>
```

### Seed your real docs

```python
from adapter import ChatMemoryAdapter

adapter = ChatMemoryAdapter()
await adapter.ingest_docs(site_id="acme", documents=["...your docs text..."])
```

## HTTP API

| Method | Path          | Body                                                        | Returns                          |
| ------ | ------------- | ----------------------------------------------------------- | -------------------------------- |
| POST   | `/api/chat`   | `{message, conversation_id, visitor_id, site_id, opt_in}`   | `{answer, citations, session_id}`|
| POST   | `/api/forget` | `{conversation_id, visitor_id, site_id}`                    | `{cleared, session_id}`          |
| GET    | `/`           | —                                                           | demo "ask our docs" page         |
| GET    | `/widget.js`  | —                                                           | the embeddable snippet           |

## Tests (deterministic, no API keys)

Memory calls are mocked, so tests run fast in CI with no real LLM:

```bash
uv run pytest cognee/tests/unit/bots/test_web_widget_bot.py -v
```

[#3608]: https://github.com/topoteretes/cognee/issues/3608
