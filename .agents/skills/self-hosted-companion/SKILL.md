---
name: self-hosted-companion
description: Chat with a companion that knows your notes folder (a journal, an Obsidian vault, any .md or .txt files) and every earlier chat, using cognee memory with local databases. Use when someone wants to ask questions about their own notes, or to talk with a companion that remembers past conversations. Runs locally; reads the notes folder, writes only to cognee's memory.
---

# Self-hosted AI companion

Runs the self-hosted companion cookbook in `examples/cookbooks/self_hosted_companion/`. It
remembers a notes folder in the cognee dataset `self_hosted_companion`, then answers
messages about it; each chat is saved to memory, so a later one knows it. Setup, the
scripts and what each one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/self_hosted_companion/README.md).

Run every command from the repo root.

## Steps

Without a notes folder of the user's own, or to see it work first, leave out `<folder>`
below: the run then uses sample notes (the `[setup]` line says so). It needs only
`LLM_API_KEY`. Tell the user the answer comes from sample notes.

1. Run
   `uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --check <folder>`
   with the notes folder the user named.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Tell the user which line to fix,
     using the README's "What it needs" table, and stop.
2. You can't type into the interactive chat, so pass the user's message with `--ask`:
   `uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py <folder> --ask "..."`.
   Once the folder is remembered, later messages only need
   `uv run python examples/cookbooks/self_hosted_companion/scripts/chat.py --ask "..."`.
3. The answer is everything after `[chat] companion>`. Give it to the user. Each `--ask` is
   its own chat session, saved to memory, so a later one knows it.
4. If a script fails, `self_hosted_companion.py` exits 1 with a line naming what went
   wrong. Run that script from `examples/cookbooks/self_hosted_companion/scripts/` on its
   own to look closer.

## Rules

- The notes are the user's private files. Don't quote them beyond what answers the
  question, and don't print `.env`.
- Every run uses LLM credits, so run it only when the user asked for it.
- A sample run forgets the cookbook's dataset `self_hosted_companion` before it ingests (the `[clear]`
  line). On the user's own data, pass `--clear` only when they asked to start over: it
  forgets everything the dataset holds, including the earlier chats.
- Don't pass `--ui` unless the user asked to browse the graph: it keeps running until
  Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset self_hosted_companion
```
