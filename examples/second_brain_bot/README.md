# Second Brain Bot

One memory graph, reachable from many front-ends. Capture a note in Telegram,
recall it from the web, and both resolve to the same private brain because the
bot knows they are the same person. Capture anywhere, recall anywhere, forget
everywhere.

This is the proof bot for issue #3613, built on the shared chat-memory adapter
contract from #3608. The memory primitives (remember, recall, forget) come from
cognee. The piece this bot adds is the cross-transport identity layer that maps
many external identities onto one canonical user and one brain.

## Run your own in 5 minutes

You do not need any API key to try the bot. The first path below runs the whole
thing, identity, routing, citations, and forget, against an in-memory adapter.

### 1. Get the code and install

```bash
git clone https://github.com/topoteretes/cognee.git
cd cognee/examples/second_brain_bot
pip install fastapi uvicorn httpx
```

Prerequisites: Python 3.10 to 3.14.

### 2. Start the bot (no key needed)

```bash
USE_FAKE_ADAPTER=true python run.py
```

You will see the web transport come up on `http://0.0.0.0:8080/message`.

### 3. Capture a note, then recall it

Save a note as user `alice`:

```bash
curl -s localhost:8080/message \
  -H 'content-type: application/json' \
  -d '{"user": "alice", "text": "I parked the car on level 3 of the garage"}'
# {"reply":"Saved to your brain."}
```

Ask for it back (end with a question mark to recall):

```bash
curl -s localhost:8080/message \
  -H 'content-type: application/json' \
  -d '{"user": "alice", "text": "where did I park?"}'
# {"reply":"From your memory: I parked the car on level 3 of the garage\n\nSources:\n  from your web note on 2026-06-12: I parked the car on level 3 of the garage (web://alice)"}
```

That is capture and cited recall through one brain. Send `/help` as the text to
see every command.

### 4. Add a second transport (Telegram)

Create a bot with [@BotFather](https://t.me/BotFather), copy the token, and
restart with it set:

```bash
USE_FAKE_ADAPTER=true TELEGRAM_BOT_TOKEN=your_token python run.py
```

Now link the two front-ends so they share one brain:

1. In Telegram, send `/link`. The bot replies with a short code.
2. Hit the web endpoint with `/link <that code>` as the text.

From now on a note saved in Telegram is recalled from the web, and the other
way round. Send `/forget me` from either side to wipe the whole brain
everywhere.

## Real cognee-backed memory

The fake adapter proves the flow with substring recall. For real semantic
memory across sessions, switch to the cognee adapter:

```bash
pip install cognee fastembed
cp .env.example .env        # then put your LLM key in .env
python run.py               # no USE_FAKE_ADAPTER
```

`.env.example` is set up for a Groq LLM plus local fastembed embeddings (no
embedding API key, no rate limits), which is the stack this adapter was
validated against. Defaults to Groq plus local fastembed for a keyless,
cost-free first run; any cognee-supported provider (including OpenAI, cognee's
own default) works by editing `.env`.

The bot stores notes in a per-user cognee dataset (`brain:{user}`) and recalls
them with graph completion, so cross-source questions ("connect the note I saved
in Telegram with the one from web") run as a real multi-hop traversal.

Ingestion runs in the background: `remember(run_in_background=True)` returns a
fast "Saved" while cognee builds the graph. A freshly saved note becomes
recallable once that build completes (tens of seconds for a short note), then it
is durably recallable from any transport. The adapter sets `ENABLE_BACKEND_ACCESS_CONTROL=false`
and `CACHING=false` before importing cognee (see `config.py`); these must be set
before the import or cognee defaults to multi-user auth.

## Verify it offline, no keys

The bot logic has a deterministic test suite that runs fully offline against
the in-memory adapter, no cognee and no API key:

```bash
pip install pytest
python -m pytest tests/
```

The tests cover cross-transport recall with a citation, identity linking
merging two front-ends, and `/forget me` wiping across both transports.

## Commands

```
send any note            remember it
ask a question (end ?)   recall it, with citations
/link                    issue a code to connect another app to this brain
/link <code>             enter a code from your other app to share one brain
/forget me               wipe your whole brain across every app
/optout                  pause capturing new notes
/optin                   resume capturing new notes
/help                    show the command list
```

## How it works

```
transports/   telegram + web. Each normalizes a platform event to a Conversation. Thin.
identity/     link table (external identity -> canonical user) + one-time-code linking.
adapter/      the #3608 contract: scope / ingest / answer / forget.
                interface.py     the contract
                fake_adapter.py  in-memory, for tests and the no-key run
                cognee_adapter.py real impl over cognee
bot/          router (resolve identity, route capture vs recall), commands, consent.
```

Memory boundary: the dataset is keyed by the canonical user, `brain:{user}`, so
a note from any transport lands in one shared brain. Durable recall targets the
whole brain, which is the persistence-across-sessions story this bot
demonstrates.

A note on `session`: the `Scope` contract carries a per-transport
`session = {transport}:{source}` field (the shape aligned with #3608), but this
reference adapter ingests dataset-only and does not write to the session cache.
Under access-control-off in a single-user config, cognee's session-to-graph
distillation bridge returns a 422 (the background improve task runs as a user
without write access to `brain:...`), so a session-ingested note never reaches
the durable graph. Dataset-only ingest is what makes cross-transport recall
work here. A session-cache-backed adapter (CACHING on, or the merged #3608
adapter) would use the `session` field for fast recent context.

Citations: the adapter records a source-to-message map at ingest, so a recalled
answer can link back to the original Telegram or web message. cognee's
`include_references` grounds the answer text in addition to that map.

Forget: `/forget me` is whole-brain in this version. Because the brain is a
single per-user dataset, one `forget(dataset=brain:{user})` call wipes
everything across every transport, and the identity links are then dropped so no
transport re-attaches. Per-transport or selective forget (drop just my Telegram
notes) is intentionally deferred to a follow-up: once cognify merges facts from
different transports into shared nodes, deleting a subset safely needs
deduplication-aware deletion (remove a node only when nothing else references
it), which belongs in the #3608 adapter core. The ingest-time metadata stamp is
kept so that follow-up has what it needs.

Identity linking is the part this bot adds on top of #3608. First contact from
any transport auto-creates a canonical user and a brain. `/link` issues a
short-lived code on one front-end; entering it on another points both external
identities at the same canonical user, so they share one brain. No link means
each external identity keeps its own brain until linked.

## Relationship to #3608

Built against the #3608 three-primitive interface as a contract. The adapter
author confirmed `scope()` returns `dataset` and `session` as separate fields,
so this bot's per-user scope (`dataset=brain:{user}`,
`session={transport}:{source}`) maps to the merged adapter natively. The local
`cognee_adapter.py` is swappable for the merged #3608 adapter behind the same
interface. The ownable piece here is the cross-transport identity layer, which
#3608 does not cover.
