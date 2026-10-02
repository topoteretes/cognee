# Company brain: agent for follow-up

Turn your latest call into next steps, posted to Slack.

cognee remembers your Granola calls, Linear issues and Gmail inbox in one company dataset
(`company_brain`). Then it works out the next steps of your latest call: who owns each one,
which team it belongs to, its deadline, and whether Linear already tracks it. None of that
has to be in the call itself: the team comes from earlier calls, a deadline from an email,
a tracked issue from Linear.

Agents run this cookbook through the `follow-up-agent` skill,
[`.agents/skills/follow-up-agent/SKILL.md`](../../../../.agents/skills/follow-up-agent/SKILL.md).

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the next steps with an LLM (OpenAI by default) | `.env` at the repo root |
| `GRANOLA_API_KEY` | Reads your calls through Granola's public API. Create one in Granola's settings | `.env` at the repo root |
| `LINEAR_API_KEY` (optional) | A personal API key (Linear: Settings → Security & access). Without it, issues are skipped | `.env` at the repo root |
| `credentials.json` (optional) | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled. Without it, email is skipped | the cookbook folder, next to `follow_up_agent.py` |
| `token.json` | Written on the first Gmail run, after you consent in the browser. Scope: `gmail.readonly` | the cookbook folder, created for you |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL` (optional) | A Slack app with the `chat:write` bot scope, invited to the channel; the channel id. Without them, the steps are printed | `.env` at the repo root |
| `cognee[gmail]` | The Google client libraries, for the Gmail step | `uv sync --extra gmail` |

`credentials.json` and `token.json` are git-ignored. Never commit or print them, or the keys.

## Try it on sample data

`setup.py` writes sample calls, Linear issues and an email to `sample/` (git-ignored), in
the shape Granola, Linear and Gmail give the scripts, dated relative to today. The latest
call agrees on next steps but names no team or deadline; an earlier call, Linear and a
bank's email hold those. Only `LLM_API_KEY` is needed, and a sample run never posts to
Slack, even when Slack is set up.

```bash
uv run python examples/cookbooks/company_brain/follow_up_agent/setup.py
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --sample
```

```text
[ingest_granola] Remembered 2 sample calls
[ingest_linear] Remembered 2 sample Linear issues
[ingest_email] Remembered 1 sample inbox emails
[follow_up] Call: Checkout v2 launch readiness
[follow_up] Steps (a sample run, so not posted):
*Next steps from "Checkout v2 launch readiness"*
- Migrate card payments to 3DS2 — Owner: Omar Haddad; Team: Payments; Due date: 2026-11-01 (Kestrel Bank requirement); Linear issue: PAY-104.
- Load-test the checkout API at 3× peak traffic — Owner: Sam Okoro; Team: Platform; Due date: 2026-10-12; Linear issue: PLAT-88.
- Write the launch announcement (after the migration and load test) — Owner: Lena Fischer; Team: Payments; Due date: not specified; Linear issue: none.
```

Omar's team comes from the earlier call, PAY-104 from Linear, and the 3DS2 deadline from
the bank's email: none of them is in the call itself.

## Steps

```
follow_up_agent/
├── README.md             this file
├── follow_up_agent.py    checks setup, then calls the scripts in order
├── setup.py              writes the sample, for --sample
├── sample/               written by setup.py, git-ignored
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
| 1 | `scripts/ingest_granola.py [--days N] [--sample]` | Remembers Granola calls from the last 30 days by default (node set `calls`) | cognee dataset |
| 2 | `scripts/ingest_linear.py [--days N] [--sample]` | Remembers Linear issues changed in the last 30 days by default (node set `linear`) | cognee dataset |
| 3 | `scripts/ingest_email.py [--emails N] [--sample]` | Remembers the newest 50 inbox emails by default, through cognee's Gmail connector `gmail_source` (node set `email`) | cognee dataset |
| 4 | `scripts/follow_up.py [--days N] [--sample]` | Fetches the latest call and the emails about it (a `CHUNKS` recall over `email`), asks a `GRAPH_COMPLETION` recall over the whole graph for its next steps, and posts them to Slack with `chat.postMessage` (or prints them) | a Slack message, when set up |
| 5 | `scripts/ui.py` (or `--ui`) | Starts cognee's API server in this process and the UI at http://localhost:3000. Ctrl+C stops both | nothing |

All scripts use the cognee dataset `company_brain`, named once in each script.

## Run it

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

## Clean up

```bash
uv run cognee-cli forget --dataset company_brain
```

The `multi_source/` cookbook writes to the same `company_brain` dataset, so this also
removes what it remembered.
