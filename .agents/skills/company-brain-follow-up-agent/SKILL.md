---
name: company-brain-follow-up-agent
description: Turn your latest Granola call into next steps (owner, team, due date, and the Linear issue that already tracks each one) and post them to Slack, using cognee memory of your calls, Linear issues and Gmail inbox. Use when someone asks for the next steps or action items of their latest call. Runs locally; reads Gmail read-only; posts to Slack only when Slack is set up.
---

# Company brain: agent for follow-up

Runs the follow-up agent cookbook in `examples/cookbooks/company_brain/follow_up_agent/`.
It remembers your Granola calls, Linear issues and Gmail inbox in the cognee dataset
`follow_up_agent`, then works out the next steps of your latest call and posts them to
Slack, or prints them. Setup, the scripts and what each one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/company_brain/follow_up_agent/README.md).

Run every command from the repo root.

## Steps

With none of Granola, Linear and Gmail set up, the commands below run on sample calls,
issues and email (the `[setup]` line says so), follow up a fictional call and never post to
Slack; only `LLM_API_KEY` is needed. Tell the user the steps come from sample data. Add
`--sample` to use the sample even when their accounts are set up.

1. Run `uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --check`.
   - Exit 0: go to step 2. `[setup] SKIPPED:` lines name optional sources that won't be
     used; mention them to the user, but they don't block the run.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the README's "What it needs" table,
     and stop.
2. Run `uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py`.
   Add `--days N` if the user named a time range, and `--no-linear` / `--no-email` if they
   don't want a source used.
3. The steps are everything after the `[follow_up] Posted to Slack:` or `[follow_up] Steps`
   line. Give them to the user, say which call they come from (the `[follow_up] Call:`
   line), which sources went in (the `[ingest_*]` lines), and whether they were posted.
4. If a script fails, `follow_up_agent.py` exits 1 with a line naming what went wrong. Run
   that script from `examples/cookbooks/company_brain/follow_up_agent/scripts/` on its own
   to look closer.

## Rules

- Gmail access is read-only.
- With Slack set up, a run posts to the user's real channel, and every run reads the
  user's real calls, issues and mailbox and uses LLM credits, so run it only when the user
  asked for it.
- Don't print the contents of `credentials.json`, `token.json` or `.env`.
- A sample run forgets the cookbook's dataset `follow_up_agent` before it ingests (the `[clear]`
  line). On the user's own data, pass `--clear` only when they asked to start over: it
  forgets everything the dataset holds.
- Don't pass `--ui` unless the user asked to browse the graph: it keeps running until
  Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset follow_up_agent
```
