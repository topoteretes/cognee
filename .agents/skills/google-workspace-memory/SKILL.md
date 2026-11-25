---
name: google-workspace-memory
description: Remember a user's Google Calendar, Gmail and Drive in one cognee memory and answer questions across them, by default preparing their client meetings of the coming week (which orders are late or at risk, why, and what to say). Use when someone asks to connect, ingest or sync their Google Workspace, or asks a question that needs their calendar, email and Drive together. Runs locally; reads Google read-only.
---

# Google Workspace memory

Runs the Google Workspace memory cookbook in `examples/cookbooks/google_workspace_memory/`.
It remembers the user's primary calendar, newest inbox emails and one Drive folder in the
cognee dataset `google_workspace_memory`, then answers one question across them. Setup, the
scripts and what each one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/google_workspace_memory/README.md).

Run every command from the repo root.

## Steps

Without `credentials.json` in the cookbook folder, the commands below run on a sample
workspace (the `[setup]` line says so) and answer about a fictional ceramics business; only
`LLM_API_KEY` is needed. Tell the user the answer is about sample data. Add `--sample` to
use the sample even when their account is set up. The sample lives in its own dataset,
`google_workspace_memory_sample`, and never mixes with the user's data.

1. Run `uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --check`.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the README's "What it needs" table,
     and stop. If only `GOOGLE_DRIVE_FOLDER_ID` is missing, ask whether to run with
     `--no-drive` instead.
2. Run `uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py`.
   Add `--ask "..."` with the user's question if they asked one, and `--no-calendar`,
   `--no-gmail` or `--no-drive` for any app they don't want used. The first run opens the
   browser once per app for consent; tell the user to approve each one.
3. The answer is everything after the line `[ask] Answer:`. Give it to the user, and say
   which sources went in (the `[ingest_calendar]`, `[ingest_gmail]` and `[ingest_drive]`
   lines).

If the user runs cognee as a server with their Google account connected there, add `--api`
to both commands: the server syncs and answers, and `--check` reports each app that is not
connected yet (the README's "Through a cognee server" section says how to connect it).

If a run stops after `--check` passed, the last `[<step>]` line (or the traceback) says why.
Tell the user what failed; rerun only that step's script from `scripts/` once it is fixed.

For a follow-up question on the same data, run
`uv run python examples/cookbooks/google_workspace_memory/scripts/ask.py --ask "..."`, with
`--sample` if the first run was on the sample; it doesn't ingest again.

## Rules

- Google access is read-only: nothing is created, changed or sent.
- Don't print the contents of `credentials.json`, the `*_token.json` files or `.env`.
- A sample run forgets the sample's dataset `google_workspace_memory_sample` before it
  ingests (the `[clear]` line). On the user's own data (`google_workspace_memory`), pass `--clear` only when they asked to
  start over: it forgets everything the dataset holds.
- Ingesting reads the user's real calendar, mailbox and Drive files, and every run uses LLM
  credits. The first run cognifies the whole primary calendar, which can take ten minutes
  or more, so tell the user before you start it and run it only when they asked for it.

## Clean up

```bash
uv run cognee-cli forget --dataset google_workspace_memory
uv run cognee-cli forget --dataset google_workspace_memory_sample
```
