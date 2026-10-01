---
name: inbox-qa
description: Answer a question from your newest Gmail email plus a text file of notes, using cognee memory. Use when someone asks what their latest email wants, or a question that needs both that email and their notes. Runs locally; reads Gmail read-only; never sends mail.
---

# Inbox Q&A

Remembers the newest email in your Gmail inbox and a text file of your notes in one cognee
dataset (`inbox_qa_skill`), then answers a question from both. Use it to check how
cognee combines two sources, or as a template for a skill with several steps.

This file is for both readers. A person can follow it top to bottom. An agent should
follow **For agents** and run the commands as written.

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the answer with an LLM (OpenAI by default) | `.env` at the repo root |
| `credentials.json` | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled in Google Cloud | this folder |
| `token.json` | Written on the first run, after you consent in the browser. Scope: `gmail.readonly` | this folder, created for you |
| `cognee[gmail]` | The Google client libraries | `uv sync --extra gmail` |

No Gmail? Run with `--no-email`; then only `LLM_API_KEY` is needed.
`credentials.json` and `token.json` are git-ignored. Never commit or print them.

## Steps

Each step is its own script and also runs on its own. `run.py` runs them in order.

| # | Script | Does | Writes |
|---|---|---|---|
| 0 | `run.py --check` | Reports what is missing. Does no work | nothing |
| 1 | `ingest_email.py` | Remembers the newest inbox email (node set `email`) | cognee dataset |
| 2 | `ingest_file.py [path]` | Remembers a text file, `data/notes.txt` by default (node set `notes`) | cognee dataset |
| 3 | `answer.py "question"` | Answers from what steps 1 and 2 remembered | nothing |

## Run it (people)

From the repo root:

```bash
uv run python examples/cookbooks/inbox_qa_skill/run.py --check
uv run python examples/cookbooks/inbox_qa_skill/run.py
uv run python examples/cookbooks/inbox_qa_skill/run.py --file my_notes.txt --question "What do I owe Priya?"
```

The first Gmail run opens a browser to consent. Running it again re-remembers the same
content; cognee skips content it already holds.

## For agents

1. Run `uv run python examples/cookbooks/inbox_qa_skill/run.py --check` from the repo root.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the table above, and stop. If only
     the Gmail lines are missing, ask whether to run with `--no-email` instead.
2. Run `run.py` with the user's question: `--question "..."`. Add `--file <path>` if the
   user named a file, and `--no-email` if they don't want Gmail used.
3. The answer is the line starting `[answer] A:`. Give it to the user in your own words,
   and say which sources went in: the email subject from `[ingest_email]` and the file from
   `[ingest_file]`.
4. If one step fails, run that step's script on its own to see its error.

Rules: Gmail access is read-only and nothing is sent. Don't print the contents of
`credentials.json`, `token.json` or `.env`. Step 1 reads the user's real mailbox and every
run uses LLM credits, so run it only when the user asked for it.

## Clean up

```bash
uv run cognee-cli forget --dataset inbox_qa_skill
```
