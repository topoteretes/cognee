---
name: inbox-qa
description: Answer a question about one of your newest Gmail emails (the newest, or the newest from a named sender; 20 are loaded by default) plus your Granola meeting notes, and draft replies in your own style learned from your sent mail, using cognee memory. Use when someone asks what their latest email wants, asks to answer or reply to it or to an email from a named person, or asks a question that needs both that email and their meetings. Runs locally; reads Gmail read-only; never sends mail.
---

# Inbox Q&A

Remembers the newest emails in your Gmail inbox (20 by default), your last 20 sent emails
and your Granola meeting notes in one cognee dataset (`inbox_qa_skill`), then answers a
question about one of those emails: the newest, or the newest from a sender you name. Facts
come from the email and the meetings; the sent emails are a sample of how you write, so a
drafted reply sounds like you. Use it to check how cognee combines two sources, or as a
template for a skill with several steps.

This file is for both readers. A person can follow it top to bottom. An agent should
follow **For agents** and run the commands as written.

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the answer with an LLM (OpenAI by default) | `.env` at the repo root |
| `credentials.json` | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled in Google Cloud | the skill folder, next to `SKILL.md` |
| `token.json` | Written on the first run, after you consent in the browser. Scope: `gmail.readonly` | the skill folder, created for you |
| `GRANOLA_API_KEY` | Reads your meeting notes through Granola's public API. Create one in Granola's settings | `.env` at the repo root |
| `cognee[gmail]` | The Google client libraries | `uv sync --extra gmail` |

No Gmail or no Granola? Run with `--no-email` or `--no-granola` to skip that step.
`credentials.json` and `token.json` are git-ignored. Never commit or print them, or the key.

## Steps

```
inbox_qa/
├── SKILL.md              this file
├── inbox_qa.py           checks setup, then calls the three scripts in order
├── credentials.json      yours, git-ignored
├── token.json            yours, git-ignored, written on the first Gmail run
└── scripts/
    ├── ingest_email.py
    ├── ingest_granola.py
    └── answer.py
```

`inbox_qa.py` imports each script and calls its function in one process. Each script also runs
alone with the same options.

| # | Command (`uv run python examples/cookbooks/inbox_qa/...`) | Does | Writes |
|---|---|---|---|
| 0 | `inbox_qa.py --check` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/ingest_email.py [--emails N]` | Remembers the `N` newest inbox emails, 20 by default (node set `email`), and your last 20 sent emails (node set `sent`) | cognee dataset |
| 2 | `scripts/ingest_granola.py [--days N]` | Remembers Granola meeting notes from the last 30 days by default (node set `meetings`) | cognee dataset |
| 3 | `scripts/answer.py [--sender NAME] "question"` | Picks the newest email (from `NAME` if given) and answers about it; a drafted reply copies the style of your sent emails | nothing |

All scripts write to the cognee dataset `inbox_qa_skill`, named once in each script.

## Run it (people)

From the repo root:

```bash
uv run python examples/cookbooks/inbox_qa/inbox_qa.py --check
uv run python examples/cookbooks/inbox_qa/inbox_qa.py
uv run python examples/cookbooks/inbox_qa/inbox_qa.py --days 7 --question "What do I owe Priya?"
uv run python examples/cookbooks/inbox_qa/inbox_qa.py --sender Priya   # draft a reply to Priya's email
uv run python examples/cookbooks/inbox_qa/inbox_qa.py --emails 50 --sender Priya   # look further back
```

The first Gmail run opens a browser to consent. Running it again re-remembers the same
content; cognee skips content it already holds.

## For agents

1. Run `uv run python examples/cookbooks/inbox_qa/inbox_qa.py --check` from the repo root.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the table above, and stop. If only
     the Gmail lines are missing, ask whether to run with `--no-email` instead.
2. Run `inbox_qa.py` with the user's question: `--question "..."`. For "answer my latest
   email", omit `--question`: the default drafts a reply. For "answer NAME" or "reply to
   NAME's email", add `--sender NAME` (matched against the From line, name or address;
   only the newest `--emails N` inbox emails are searched, 20 by default; raise it if the
   sender isn't found). Add `--days N` if the user named a time range for meetings, and
   `--no-email` / `--no-granola` if they don't want a source used.
3. The answer is everything from the line starting `[answer] A:` to the end. Give it to
   the user in your own words, and say which sources went in: the email answered from
   `[answer] Email:`, the sent-mail count from `[ingest_email]` and the meeting count from
   `[ingest_granola]`. A drafted reply is never sent: hand it to the user to send.
4. If a script fails, `inbox_qa.py` exits 1 with a line naming what went wrong. Run that script
   on its own to look closer.

Rules: Gmail access is read-only and nothing is sent. Don't print the contents of
`credentials.json`, `token.json` or `.env`. Steps 1 and 2 read the user's real mailbox and
meetings and every run uses LLM credits, so run them only when the user asked for it.

## Clean up

```bash
uv run cognee-cli forget --dataset inbox_qa_skill
```
