# Company brain: agent for follow-up

Turn your latest call into next steps, posted to Slack.

cognee remembers your Granola calls, Linear issues and Gmail inbox into one company graph.
Then it works out the next steps of your latest call: who owns each one, which team it
belongs to, its deadline, and whether Linear already tracks it. None of that has to be in the
call itself: the team comes from earlier calls, a deadline from an email, a tracked issue
from Linear.

## Set it up

| | |
|---|---|
| LLM | `LLM_API_KEY` in `.env`. |
| Granola | Create an API key in Granola and set `GRANOLA_API_KEY`. The last 30 days of calls are remembered. |
| Linear (optional) | Create a personal API key (Settings → Security & access) and set `LINEAR_API_KEY`. Issues changed in the last 30 days are remembered. |
| Gmail (optional) | `uv sync --extra gmail`, then save a Desktop-app OAuth client as `credentials.json` in this folder (see `examples/guides/gmail.py`). |
| Slack (optional) | A Slack app with the `chat:write` bot scope, invited to the channel. Set `SLACK_BOT_TOKEN` and `SLACK_CHANNEL` (the channel id). Without them the steps are printed. |

## Run it

```bash
uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py
```

The output looks like this (an illustration; yours comes from your own calls):

```text
Remembering your Granola calls...
Remembering your Linear issues...
Remembering your Gmail inbox...
Posted to Slack:
*Next steps from "Checkout v2 launch readiness"*
1. Migrate card payments to 3DS2 (Omar Haddad, Payments, due 2026-11-15)
...
```

## How it works

1. `cognee.remember` stores each source in its own node set: `calls`, `linear` and `email`.
   Gmail comes in through cognee's Gmail connector, `gmail_source`.
2. A `GRAPH_COMPLETION` recall with the latest call and a next-steps prompt answers from the
   whole graph, so owners, teams, deadlines and tracked issues come from every source.
3. The answer is posted to Slack with `chat.postMessage`.

Add `--ui` to the command to browse the graph afterwards: the script starts cognee's API
server in its own process, next to the databases it has open, and the UI at
http://localhost:3000. Ctrl+C stops both.
