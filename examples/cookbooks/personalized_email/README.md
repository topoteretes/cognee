# Personalized email with cognee

Draft email replies that already know what you discussed in meetings, what you promised, and
how you write.

cognee remembers your Granola meeting notes and your Gmail inbox and sent mail into one
knowledge graph. Then it drafts a reply to the newest email in your inbox: the facts come
from memory, and the tone from your own sent mail. The draft is printed, never sent.

## Set it up

| | |
|---|---|
| LLM | `LLM_API_KEY` in `.env`. |
| Gmail | `uv sync --extra gmail`. In Google Cloud, enable the Gmail API and create an OAuth client of type *Desktop app*. Save its JSON as `credentials.json` in this folder. The first run opens a browser to consent. |
| Granola (optional) | Create an API key in Granola and set `GRANOLA_API_KEY`. The last 30 days of notes are remembered. |
| Your name | Set `MY_NAME` to your name as it appears in your email. |

## Run it

```bash
uv run python examples/cookbooks/personalized_email/personalized_email.py
```

The output looks like this (an illustration; yours comes from your own mail):

```text
Remembering 12 Granola meeting notes...
Remembering your Gmail inbox...
Remembering your Gmail sent_mail...

== Incoming ==
Subject: Pilot start and SSO
...

== Draft reply ==
Hi Priya, ...
```

## How it works

1. `cognee.remember` stores each source in its own node set: `meetings`, `inbox` and
   `sent_mail`. Gmail comes in through cognee's Gmail connector, `gmail_source`. To keep the
   first try small it loads the newest 50 messages of each label.
2. A `CHUNKS` recall over the `sent_mail` node set returns a few of your own emails, for tone.
3. A `GRAPH_COMPLETION` recall writes the reply from the graph: who the sender is, what was
   decided in your meetings, and what you promised them.

Add `--ui` to the command to browse the graph afterwards: the script starts cognee's API
server in its own process, next to the databases it has open, and the UI at
http://localhost:3000. Ctrl+C stops both.
