# Company brain: agent for follow-up

After every call, an agent works out the next steps and asks in Slack whether it got them
right. Once someone confirms, it files the Linear issues.

Granola calls, Gmail and Linear issues are remembered into one company graph. For each new
call the agent asks memory who the attendees are, which team and project each next step
belongs to, which deadlines apply, and which steps Linear already tracks. It then posts to
Slack:

```text
Follow-up for "Checkout v2 launch readiness"
I think these are the next steps:
1. Migrate card payments to 3DS2 (Omar Haddad · Payments · 2026-11-15)
2. Own Checkout v2 error-rate dashboard (Omar Haddad · Payments) — already tracked in PAY-104
3. Write launch comms plan for Support and Sales (Nina Park · Payments · 2026-10-14)
Reply yes in this thread, or yes 1 3 for some of them, or react with ✅, and I'll create
the Linear issues. Reply no or react with ❌ to skip.
```

None of those details are in the call alone:
- The Nov 15 due date comes from an email from Adyen.
- The Payments team comes from earlier notes.
- PAY-104 comes from Linear.

After a *yes*, the agent creates issues 1 and 3, replies in the thread with their links, and
remembers that they now exist.

## Run it on the sample data

Needs `LLM_API_KEY` in `.env`.

```bash
# 1. Remember the sample call, emails and Linear issues
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py ingest --sample

# 2. A new call "arrives": propose its next steps, printing the Slack message and the
#    Linear issues instead of sending them
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py follow-up --sample --dry-run
```

Once `SLACK_BOT_TOKEN` and `SLACK_CHANNEL` are set, drop `--dry-run` to post the sample
proposal to your channel. Answer it in Slack, then run `follow-up --sample` again, or keep it
running with `--watch`. Your answer is picked up and the issues are created once
`LINEAR_API_KEY` is set too.

## Run it on your own data

| Service | Setup |
|---|---|
| Granola | Create an API key in Granola and set `GRANOLA_API_KEY`. |
| Linear | Create a personal API key (Settings → Security & access) and set `LINEAR_API_KEY`. Set `LINEAR_DEFAULT_TEAM` to a team name or key for steps whose team memory does not know. |
| Gmail | `uv sync --extra gmail`, then save a Desktop-app OAuth client as `credentials.json` in this folder (see `examples/guides/gmail.py`). |
| Slack | Create a Slack app with the bot scopes `chat:write`, `channels:history` (`groups:history` for a private channel) and `reactions:read`. Install it, invite the bot to the channel, and set `SLACK_BOT_TOKEN` and `SLACK_CHANNEL` (the channel id). |

```bash
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py ingest
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py serve --interval 300
```

`ingest` treats calls that are already in Granola as history. The agent follows up only on
calls that arrive after it.

## How it works

| Step | Where |
|---|---|
| **Ingest.** Calls, email and Linear issues go into their own node sets (`granola_calls`, `email`, `linear_issues`) in one dataset. | `remember_batch` |
| **Graph model.** `Person`, `Team`, `Project`, `Issue`, `Meeting` and `ActionItem`, each with `identity_fields`. An `Issue` is identified by its Linear identifier, which is how the agent recognizes a step that is already tracked. | `models.py` |
| **Keep it live.** Granola and Linear keep watermarks in `.state.json` (newest note, newest issue update). Gmail uses cognee's connector, which tracks its own position. | `sync_once` |
| **UI.** `serve` runs the cognee API in the same process as the agent loop, because the embedded graph database allows one process at a time. Then it starts the UI. | `serve` |
| **Follow-up agent.** A graph recall answers who the attendees are: team, role, projects. Chunk recalls over the `linear_issues` and `email` node sets return the raw text closest to the call, so issue identifiers and deadlines are copied exactly. One structured LLM call turns the call plus that context into next steps. The agent then asks in Slack and waits: proposals are stored, and each run checks the thread and reactions. Confirmed steps become Linear issues and are remembered. | `extract_next_steps`, `follow_up_once`, `actions.py` |

Slack's interactive buttons would need a public URL, so the agent reads plain thread replies
and reactions instead. That way it runs on a laptop.

Memory is stored in `.cognee_system/` in this folder, and `ingest` rebuilds it from scratch.
