---
name: follow-up-agent
description: Turn your latest Granola call into next steps (owner, team, due date, and the Linear issue that already tracks each one) and post them to Slack, using cognee memory of your calls, Linear issues and Gmail inbox. Use when someone asks for the next steps or action items of their latest call. Runs locally; reads Gmail read-only; posts to Slack only when Slack is set up.
---

# Company brain: agent for follow-up

Turn your latest call into next steps, posted to Slack.

cognee remembers your Granola calls, Linear issues and Gmail inbox in one company dataset
(`company_brain`). Then it works out the next steps of your latest call: who owns each one,
which team it belongs to, its deadline, and whether Linear already tracks it. None of that
has to be in the call itself: the team comes from earlier calls, a deadline from an email,
a tracked issue from Linear.

This file is for both readers. A person can follow it top to bottom. An agent should
follow **For agents** and run the commands as written.

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the next steps with an LLM (OpenAI by default) | `.env` at the repo root |
| `GRANOLA_API_KEY` | Reads your calls through Granola's public API. Create one in Granola's settings | `.env` at the repo root |
| `LINEAR_API_KEY` (optional) | A personal API key (Linear: Settings → Security & access). Without it, issues are skipped | `.env` at the repo root |
| `credentials.json` (optional) | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled. Without it, email is skipped | the cookbook folder, next to `SKILL.md` |
| `token.json` | Written on the first Gmail run, after you consent in the browser. Scope: `gmail.readonly` | the cookbook folder, created for you |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL` (optional) | A Slack app with the `chat:write` bot scope, invited to the channel; the channel id. Without them, the steps are printed | `.env` at the repo root |
| `cognee[gmail]` | The Google client libraries, for the Gmail step | `uv sync --extra gmail` |

`credentials.json` and `token.json` are git-ignored. Never commit or print them, or the keys.

## Steps

```
follow_up_agent/
├── SKILL.md              this file
├── follow_up_agent.py    checks setup, then calls the scripts in order
├── credentials.json      yours, git-ignored (optional)
├── token.json            yours, git-ignored, written on the first Gmail run
└── scripts/
    ├── ingest_granola.py
    ├── ingest_linear.py
    ├── ingest_email.py
    ├── follow_up.py
    └── ui.py
```

`follow_up_agent.py` imports each script and calls its function in one process. Each
script also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/company_brain/follow_up_agent/...`) | Does | Writes |
|---|---|---|---|
| 0 | `follow_up_agent.py --check` | Reports what is missing and which optional sources are skipped. Does no work | nothing |
| 1 | `scripts/ingest_granola.py [--days N]` | Remembers Granola calls from the last 30 days by default (node set `calls`) | cognee dataset |
| 2 | `scripts/ingest_linear.py [--days N]` | Remembers Linear issues changed in the last 30 days by default (node set `linear`) | cognee dataset |
| 3 | `scripts/ingest_email.py [--emails N]` | Remembers the newest 50 inbox emails by default, through cognee's Gmail connector `gmail_source` (node set `email`) | cognee dataset |
| 4 | `scripts/follow_up.py [--days N]` | Fetches the latest call, asks a `GRAPH_COMPLETION` recall over the whole graph for its next steps, and posts them to Slack with `chat.postMessage` (or prints them) | a Slack message, when set up |
| 5 | `scripts/ui.py` (or `--ui`) | Starts cognee's API server in this process and the UI at http://localhost:3000. Ctrl+C stops both | nothing |

All scripts use the cognee dataset `company_brain`, named once in each script.

## Run it (people)

From the repo root:

```bash
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --check
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --days 7 --ui
```

The output looks like this (an illustration; yours comes from your own calls):

```text
[ingest_granola] Remembered 9 Granola calls from the last 30 days
[ingest_linear] Remembered 41 Linear issues from the last 30 days
[ingest_email] Remembered your newest 50 inbox emails
[follow_up] Call: Checkout v2 launch readiness
[follow_up] Posted to Slack:
*Next steps from "Checkout v2 launch readiness"*
1. Migrate card payments to 3DS2 (Omar Haddad, Payments, due 2026-11-15)
...
```

The first Gmail run opens a browser to consent. Running it again re-remembers the same
content; cognee skips content it already holds, and Gmail rows are merged by message id.

## For agents

1. Run `uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --check`
   from the repo root.
   - Exit 0: go to step 2. `[setup] SKIPPED:` lines name optional sources that won't be
     used; mention them to the user, but they don't block the run.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the table above, and stop.
2. Run `follow_up_agent.py`. Add `--days N` if the user named a time range, and
   `--no-linear` / `--no-email` if they don't want a source used.
3. The steps are everything after the `[follow_up] Posted to Slack:` or `[follow_up] Steps`
   line. Give them to the user, say which call they come from (the `[follow_up] Call:`
   line), which sources went in (the `[ingest_*]` lines), and whether they were posted.
4. If a script fails, `follow_up_agent.py` exits 1 with a line naming what went wrong. Run
   that script on its own to look closer.

Rules: Gmail access is read-only. With Slack set up, a run posts to the user's real
channel, and every run reads the user's real calls, issues and mailbox and uses LLM
credits, so run it only when the user asked for it. Don't print the contents of
`credentials.json`, `token.json` or `.env`. Don't pass `--ui` unless the user asked to
browse the graph: it keeps running until Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset company_brain
```

The `multi_source/` cookbook writes to the same `company_brain` dataset, so this also
removes what it remembered.
