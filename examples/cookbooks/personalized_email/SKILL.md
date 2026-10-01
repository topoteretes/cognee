---
name: personalized-email
description: Draft a reply to your newest Gmail email that knows what you discussed in your Granola meetings, what you promised the sender, and how you write, using cognee memory. Use when someone asks to answer or reply to their latest email in their own tone. Runs locally; reads Gmail read-only; the draft is printed, never sent.
---

# Personalized email

Draft email replies that already know what you discussed in meetings, what you promised,
and how you write.

cognee remembers your Granola meeting notes and your Gmail inbox and sent mail in one
dataset (`personalized_email`). Then it drafts a reply to the newest email in your inbox:
the facts come from memory, and the tone from your own sent mail. The draft is printed,
never sent.

This file is for both readers. A person can follow it top to bottom. An agent should
follow **For agents** and run the commands as written.

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the draft with an LLM (OpenAI by default) | `.env` at the repo root |
| `credentials.json` | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled in Google Cloud | the cookbook folder, next to `SKILL.md` |
| `token.json` | Written on the first run, after you consent in the browser. Scope: `gmail.readonly` | the cookbook folder, created for you |
| `GRANOLA_API_KEY` | Reads your meeting notes through Granola's public API. Create one in Granola's settings | `.env` at the repo root |
| `MY_NAME` (optional) | Your name as it appears in your email, so the draft speaks as you | `.env` at the repo root |
| `cognee[gmail]` | The Google client libraries | `uv sync --extra gmail` |

No Granola? Run with `--no-granola` to skip that step.
`credentials.json` and `token.json` are git-ignored. Never commit or print them, or the keys.

## Steps

```
personalized_email/
├── SKILL.md                this file
├── personalized_email.py   checks setup, then calls the scripts in order
├── credentials.json        yours, git-ignored
├── token.json              yours, git-ignored, written on the first Gmail run
└── scripts/
    ├── ingest_granola.py
    ├── ingest_email.py
    └── draft.py
```

`personalized_email.py` imports each script and calls its function in one process. Each
script also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/personalized_email/...`) | Does | Writes |
|---|---|---|---|
| 0 | `personalized_email.py --check` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/ingest_granola.py [--days N]` | Remembers Granola meeting notes from the last 30 days by default (node set `meetings`) | cognee dataset |
| 2 | `scripts/ingest_email.py [--emails N]` | Remembers the newest 50 inbox and 50 sent emails by default, through cognee's Gmail connector `gmail_source` (node sets `inbox`, `sent_mail`) | cognee dataset |
| 3 | `scripts/draft.py` | Drafts a reply to the newest inbox email: facts from a `GRAPH_COMPLETION` recall over the graph, tone from a `CHUNKS` recall over `sent_mail` | nothing |

All scripts use the cognee dataset `personalized_email`, named once in each script.

## Run it (people)

From the repo root:

```bash
uv run python examples/cookbooks/personalized_email/personalized_email.py --check
uv run python examples/cookbooks/personalized_email/personalized_email.py
uv run python examples/cookbooks/personalized_email/personalized_email.py --days 7
```

The output looks like this (an illustration; yours comes from your own mail):

```text
[ingest_granola] Remembered 12 Granola meetings from the last 30 days
[ingest_email] Remembered your newest 50 inbox emails
[ingest_email] Remembered your newest 50 sent emails
[draft] Answering: Pilot start and SSO (from Priya Shah <priya@northwind.example>)
[draft] Reply:

To: Priya Shah <priya@northwind.example>
Subject: Re: Pilot start and SSO

Hi Priya,

...

Best,
M.
```

The first Gmail run opens a browser to consent. Running it again re-remembers the same
content; cognee skips content it already holds, and Gmail rows are merged by message id.

## For agents

1. Run `uv run python examples/cookbooks/personalized_email/personalized_email.py --check`
   from the repo root.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the table above, and stop. If only the
     Granola line is missing, ask whether to run with `--no-granola` instead.
2. Run `personalized_email.py`. Add `--days N` if the user named a time range for meetings,
   and `--no-granola` if they don't want Granola used.
3. The draft is everything after the line `[draft] Reply:`. Give it to the user, and say
   which email it answers (the `[draft] Answering:` line) and which sources went in (the
   `[ingest_granola]` and `[ingest_email]` lines). The draft is never sent: hand it to the
   user to send.
4. If a script fails, `personalized_email.py` exits 1 with a line naming what went wrong.
   Run that script on its own to look closer.

Rules: Gmail access is read-only and nothing is sent. Don't print the contents of
`credentials.json`, `token.json` or `.env`. Steps 1 and 2 read the user's real meetings and
mailbox and every run uses LLM credits, so run them only when the user asked for it.

## Clean up

```bash
uv run cognee-cli forget --dataset personalized_email
```
