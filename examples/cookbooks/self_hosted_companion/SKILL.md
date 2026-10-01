---
name: self-hosted-companion
description: Chat with a companion that knows your notes folder (a journal, an Obsidian vault, any .md or .txt files) and every earlier chat, using cognee memory with local databases. Use when someone wants to ask questions about their own notes, or to talk with a companion that remembers past conversations. Runs locally; reads the notes folder, writes only to cognee's memory.
---

# Self-hosted AI companion

A chat companion that knows your notes.

cognee remembers your notes folder (a journal, an Obsidian vault, any `.md` or `.txt`
files) in one dataset (`companion`), then you chat with it. When the chat ends, cognee
writes the chat into memory, so the next chat knows what you said.

cognee's databases are local files on your machine. The LLM and the embeddings are the ones
your `.env` configures (OpenAI by default, so set `LLM_API_KEY`).

This file is for both readers. A person can follow it top to bottom. An agent should
follow **For agents** and run the commands as written.

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and answers with an LLM (OpenAI by default) | `.env` at the repo root |
| A notes folder | The notes the companion knows: `.md` or `.txt` files, any depth | anywhere; pass its path |

## Steps

```
self_hosted_companion/
├── SKILL.md                    this file
├── self_hosted_companion.py    checks setup, then calls the scripts in order
└── scripts/
    ├── ingest_notes.py
    ├── chat.py
    └── ui.py
```

`self_hosted_companion.py` imports each script and calls its function in one process. Each
script also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/self_hosted_companion/...`) | Does | Writes |
|---|---|---|---|
| 0 | `self_hosted_companion.py --check <folder>` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/ingest_notes.py <folder>` | Remembers every note in the folder | cognee dataset |
| 2 | `scripts/chat.py [--ask "message"]` | Chats in one session: each message is a `GRAPH_COMPLETION` recall that sees the turns before it. `/bye` ends it, and `improve(session_ids=[...])` writes the chat into memory. `--ask` answers one message and saves it the same way | cognee dataset |
| 3 | `scripts/ui.py` (or `--ui`) | Starts cognee's API server in this process and the UI at http://localhost:3000. Ctrl+C stops both | nothing |

All scripts use the cognee dataset `companion`, named once in each script.

## Run it (people)

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
memory.

## For agents

1. Run `uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --check <folder>`
   from the repo root, with the notes folder the user named.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Tell the user which line to fix,
     using the table above, and stop.
2. You can't type into the interactive chat, so pass the user's message with `--ask`:
   `self_hosted_companion.py <folder> --ask "..."`. Once the folder is remembered, later
   messages only need `scripts/chat.py --ask "..."`.
3. The answer is everything after `[chat] companion>`. Give it to the user. Each `--ask` is
   its own chat session, saved to memory, so a later one knows it.
4. If a script fails, `self_hosted_companion.py` exits 1 with a line naming what went
   wrong. Run that script on its own to look closer.

Rules: the notes are the user's private files. Don't quote them beyond what answers the
question, and don't print `.env`. Every run uses LLM credits, so run it only when the user
asked for it. Don't pass `--ui` unless the user asked to browse the graph: it keeps running
until Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset companion
```
