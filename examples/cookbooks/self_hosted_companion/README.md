# Self-hosted AI companion

A chat companion that knows your notes.

cognee remembers your notes folder (a journal, an Obsidian vault, any `.md` or `.txt`
files) in one dataset (`self_hosted_companion`), then you chat with it. When the chat
ends, cognee writes the chat into memory, so the next chat knows what you said.

cognee's databases are local files on your machine. The LLM and the embeddings are the ones
your `.env` configures (OpenAI by default, so set `LLM_API_KEY`). To run them on your
machine too, see [Run fully local](#run-fully-local).

Agents run this cookbook through the `self-hosted-companion` skill,
[`.agents/skills/self-hosted-companion/SKILL.md`](../../../.agents/skills/self-hosted-companion/SKILL.md).

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and answers with an LLM (OpenAI by default) | `.env` at the repo root |
| A notes folder | The notes the companion knows: `.md` or `.txt` files, any depth | anywhere; pass its path |

## Try it on sample notes

`setup.py` writes a few journal entries to `sample/notes/` (git-ignored), dated relative to
today. Only `LLM_API_KEY` is needed.

```bash
uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py
```

With no notes folder given, the script runs `setup.py` itself and uses the sample, as below. Pass `--sample` to
use it even when your own sources are set up.
A sample run first forgets the cookbook's dataset (`self_hosted_companion`), so copies
from an earlier run (the sample notes are named after dates relative to today) never mix
with this one. `--no-clear` keeps it.

```text
[clear] Forgot the dataset self_hosted_companion
[ingest_notes] Remembered the notes in .../sample/notes
[chat] you> When is my sister's birthday, and what was I planning to get her?
[chat] companion> Her birthday is on October 4, 2026. You planned to give her a voucher for a pottery class at the ceramics studio on Linden Street.
[chat] Saved this chat to memory.
```

The sample run asks that question unless you pass `--ask`.

## Steps

```
self_hosted_companion/
├── README.md                   this file
├── self_hosted_companion.py    checks setup, then calls the scripts in order
├── setup.py                    writes the sample notes, for --sample
├── sample/                     written by setup.py, git-ignored
└── scripts/
    ├── clear.py
    ├── ingest_notes.py
    ├── chat.py
    └── ui.py
```

`self_hosted_companion.py` imports each script and calls its function in one process. Each
script also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/self_hosted_companion/...`) | Does | Writes |
|---|---|---|---|
| 0 | `self_hosted_companion.py --check [folder]` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/clear.py` (or `--clear`) | Forgets the dataset `self_hosted_companion`, with the earlier chats saved in it. Runs first with `--clear`, which is on by default for a sample run (`--no-clear` turns it off) | cognee dataset |
| 2 | `scripts/ingest_notes.py <folder>` | Remembers every note in the folder | cognee dataset |
| 3 | `scripts/chat.py [--ask "message"]` | Chats in one session: each message is a `HYBRID_COMPLETION` recall that sees the turns before it. `/bye` ends it, and `improve(session_ids=[...])` writes the chat into memory. `--ask` answers one message and saves it the same way | cognee dataset |
| 4 | `scripts/ui.py` (or `--ui`) | Starts cognee's API server in this process and the UI at http://localhost:3000. Ctrl+C stops both | nothing |

All scripts use the cognee dataset `self_hosted_companion`, named once in each script.

## Run it

From the repo root:

```bash
uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --check ~/Documents/journal
uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py ~/Documents/journal
```

```text
[ingest_notes] Remembered the notes in /Users/you/Documents/journal
[chat] Chat with your companion. Type /bye to end.
you> When is my sister's birthday?
companion> Your sister Lena's birthday is on October 4th. ...
you> /bye
[chat] Saved this chat to memory.
```

Running it again remembers the folder again: new notes are added and unchanged notes are
skipped. An edited note is remembered as a new document, and its old version stays in
memory. Add `--clear` to forget the dataset `self_hosted_companion` first and start over
from your notes as they are now; that also forgets the earlier chats.

## Run fully local

To keep your notes on your machine, point the LLM and the embeddings at
[Ollama](https://ollama.com) in `.env`. The scripts need no change.

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_ENDPOINT=http://localhost:11434      # the bare address; with /v1, calls return 404
LLM_API_KEY=ollama                       # any value; Ollama ignores it
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_ENDPOINT=http://localhost:11434/api/embed
EMBEDDING_DIMENSIONS=768
HUGGINGFACE_TOKENIZER=nomic-ai/nomic-embed-text-v1.5
AUTO_FEEDBACK=false                      # skips a second LLM call per chat turn, slow on a local model
TELEMETRY_DISABLED=1
```

Larger models extract a better graph. If you change the embedding model, set
`EMBEDDING_DIMENSIONS` to its size and run `forget` (below) first, because the stored vectors
have the old size.

## Clean up

```bash
uv run cognee-cli forget --dataset self_hosted_companion
```
