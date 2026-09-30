# Self-hosted AI companion

A chat companion that remembers your life and runs only on your machine.

Your journal notes are remembered into a small life graph: the people around you, your
goals, your preferences and the dates coming up. Every chat opens with a check-in built from
that memory. When a chat ends, cognee improves memory from it, so the next chat knows what
you said:

```text
$ companion.py chat --message "Big news: Lena adopted a beagle puppy called Pepper yesterday."
companion> That's wonderful news, Jordan! ...

$ companion.py chat --message "What is the name of my sister's dog?"
companion> Your sister's dog is named Pepper. Lena adopted a beagle puppy called Pepper recently.
```

Nothing leaves the machine:
- The LLM and the embeddings run on Ollama.
- The graph, vector and relational databases are files in this folder.
- Telemetry is off.

The script fixes the provider to Ollama, so an LLM key in `.env` is never used here.

## Run it on the sample journal

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

uv run python examples/cookbooks/self_hosted_companion/companion.py ingest --sample
uv run python examples/cookbooks/self_hosted_companion/companion.py chat
```

`chat` is interactive. Type `/bye` to end it and save it to memory.
`chat --message "..."` sends one message and exits, and `chat --session <id>` continues a
session.

A new journal entry "arrives" in `sample_data/incoming/notes`. `sync` remembers only that entry:

```bash
uv run python examples/cookbooks/self_hosted_companion/companion.py sync --sample
```

Browse the graph in the UI while the notes folder keeps syncing (the UI needs Node.js and npm,
or Docker):

```bash
uv run python examples/cookbooks/self_hosted_companion/companion.py serve --sample
```

## Use your own notes

```bash
uv run python examples/cookbooks/self_hosted_companion/companion.py ingest --notes ~/Documents/journal
uv run python examples/cookbooks/self_hosted_companion/companion.py serve --interval 60
```

Notes are `.md` or `.txt` files. `sync` remembers a note again when you edit it. Set
`COMPANION_USER_NAME` to your name.

| Variable | Default | |
|---|---|---|
| `COMPANION_LLM_MODEL` | `llama3.1:8b` | Any Ollama chat model. Larger models extract better. |
| `COMPANION_EMBEDDING_MODEL` | `nomic-embed-text` | Any Ollama embedding model. |
| `COMPANION_EMBEDDING_DIMENSIONS` | `768` | Must match the embedding model (`qwen3-embedding` is 4096). |
| `COMPANION_TOKENIZER` | `nomic-ai/nomic-embed-text-v1.5` | Hugging Face tokenizer for the embedding model, downloaded once. |
| `COMPANION_OLLAMA_URL` | `http://localhost:11434` | The Ollama server. |
| `AUTO_FEEDBACK` | `false` | Analyzes every turn with a second LLM call. Slow on a local model. |

The first call after Ollama loads a model can take longer than cognee's 30-second connection
test. If `ingest` fails with `LLM connection test timed out`, run it again once the model has
loaded.

## How it works

| Step | Where |
|---|---|
| **Ingest.** Each note file is remembered with the life graph model. | `ingest`, `remember_notes` |
| **Graph model.** `Person`, `Goal`, `Preference` and `Event`: four flat types with a few fields each, which a local 8B model fills reliably. Each type maps to one question the companion asks at a check-in. | `models.py` |
| **Keep it live.** `sync` compares each note's modification time with `.state.json` and remembers only new or edited notes. | `sync_once` |
| **UI.** `serve` runs the cognee API in the same process as the sync loop, because the embedded graph database allows one process at a time. Then it starts the UI. | `serve` |
| **Companion.** Each message is a graph recall in the chat's session, with a companion system prompt, so answers see the turns before them. When the chat ends, `cognee.improve(session_ids=[...])` writes the chat into the graph. That is how the next session knows about Pepper. | `chat`, `end_chat` |

Everything the companion knows is stored in `.cognee_system/` in this folder. Delete it, and
`.state.json`, and the companion forgets everything.
