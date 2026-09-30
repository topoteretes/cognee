# Self-hosted AI companion

A chat companion that knows your notes.

cognee remembers your notes folder (a journal, an Obsidian vault, any `.md` or `.txt` files)
into a knowledge graph, then you chat with it. When the chat ends, cognee writes the chat into
memory, so the next chat knows what you said.

cognee's databases are local files on your machine. The LLM and the embeddings are the ones
your `.env` configures (OpenAI by default, so set `LLM_API_KEY`).

## Run it

```bash
uv run python examples/cookbooks/self_hosted_companion/companion.py ~/Documents/journal
```

```text
Remembering the notes in /Users/you/Documents/journal...
Chat with your companion. Type /bye to end.
you> When is my sister's birthday?
companion> Your sister Lena's birthday is on October 4th. ...
you> /bye
Saving this chat to memory...
```

## How it works

1. `cognee.remember(notes_folder)` reads every note and builds the graph.
2. Each message is a `cognee.recall` in the chat's session, so answers see the turns before
   them.
3. `cognee.improve(session_ids=[...])` writes the chat into the graph when it ends.

Running it again remembers the folder again: new notes are added and unchanged notes are
skipped. An edited note is remembered as a new document, and its old version stays in memory.
Add `--ui` to the command to browse the graph afterwards: the script starts cognee's API
server in its own process, next to the databases it has open, and the UI at
http://localhost:3000. Ctrl+C stops both.
