# Personalized email with cognee

Draft email replies that already know what you discussed in meetings, what you promised, and
how you write.

Your Granola meeting notes and your Gmail inbox and sent mail are remembered into one graph.
When an email arrives, a draft agent asks memory four things and writes the reply from the
answers:

1. Who is the sender, and what did we discuss and decide in our meetings?
2. What did I promise them, and did I deliver it?
3. What are the answers to the questions in this email?
4. How do I write? (a few of my own sent emails)

With the sample data, Priya asks when the pilot starts and whether SSO costs extra. The
answers are only in a Granola note. The agent also notices that the security questionnaire
promised in that meeting never went out:

```text
Hi Priya,

Quick answers for Dana:
- Pilot start: October 14 (runs six weeks).
- Okta SSO: included in the Enterprise tier at no extra cost, not an add-on.

On the security questionnaire: I have not sent it yet, sorry for the delay. ...

Best,
Alex
```

## Run it on the sample data

Needs `LLM_API_KEY` in `.env`. The UI also needs Node.js and npm (or Docker).

```bash
# 1. Remember the sample meeting notes and emails
uv run python examples/cookbooks/personalized_email/personalized_email.py ingest --sample

# 2. Draft a reply to the sample email from Priya
uv run python examples/cookbooks/personalized_email/personalized_email.py draft --sample

# 3. A new meeting note and email "arrive"; sync loads only those
uv run python examples/cookbooks/personalized_email/personalized_email.py sync --sample
uv run python examples/cookbooks/personalized_email/personalized_email.py draft --sample --message-id msg_inbox_harborview_nudge

# 4. Browse the graph in the UI while it keeps syncing
uv run python examples/cookbooks/personalized_email/personalized_email.py serve --sample
```

Drafts are printed and saved to `drafts/<message id>.json`. Nothing is sent.

## Run it on your own data

| Source | Setup |
|---|---|
| Granola | Create an API key in Granola and set `GRANOLA_API_KEY`. The first sync reads the last `INGEST_SINCE_DAYS` days (default 30). |
| Gmail | `uv sync --extra gmail`. In Google Cloud, enable the Gmail API and create an OAuth client of type *Desktop app*. Save its JSON as `credentials.json` in this folder (or set `GMAIL_CREDENTIALS_PATH`). The first run opens a browser to consent. Set `GMAIL_MAX_RESULTS=50` for a quick first load. |

Set `MY_NAME` to your name, so the agent knows which emails are yours. Then:

```bash
uv run python examples/cookbooks/personalized_email/personalized_email.py ingest
uv run python examples/cookbooks/personalized_email/personalized_email.py draft --message-id <Gmail message id>
uv run python examples/cookbooks/personalized_email/personalized_email.py serve --interval 300
```

A Gmail message id is the last part of the message URL in Gmail.

## How it works

| Step | Where |
|---|---|
| **Ingest.** Each source goes into its own node set (`granola_notes`, `inbox`, `sent_mail`), so the agent can read only your sent mail when it looks for your writing style. | `ingest`, `remember` |
| **Graph model.** `Person`, `Organization`, `Meeting`, `Commitment` and `EmailThread`, each with `identity_fields`. The same person in a meeting note and in an email becomes one node. The types match the agent's questions: a `Commitment` has an owner, a recipient and a due date. | `models.py` |
| **Keep it live.** Gmail uses cognee's connector, which tracks its own position and forgets deleted messages. Granola keeps a watermark in `.state.json`. `sync --watch` or `serve` repeats this every `--interval` seconds. | `sync_once` |
| **UI.** `serve` runs the cognee API in the same process as the sync loop, because the embedded graph database allows one process at a time. Then it starts the UI. | `serve` |
| **Draft agent.** Three graph recalls in one session, so each question sees the ones before it. Then two chunk recalls that read raw text from one node set each: the meeting notes closest to the email, so dates and terms are copied exactly, and `sent_mail`, for tone. Finally one structured LLM call returns the subject, the body and the facts it used. | `draft` |

Memory is stored in `.cognee_system/` in this folder, and `ingest` rebuilds it from scratch.
Delete that folder and `.state.json` to start over.
