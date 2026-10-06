# Personalized email

Draft email replies that already know what you discussed in meetings, what you promised,
and how you write.

cognee remembers your Granola meeting notes and your Gmail inbox and sent mail in one
dataset (`personalized_email`). Then it drafts a reply to the newest email in your inbox:
the facts come from memory, and the tone from your own sent mail. The draft is printed,
never sent.

Agents run this cookbook through the `personalized-email` skill,
[`.agents/skills/personalized-email/SKILL.md`](../../../.agents/skills/personalized-email/SKILL.md).

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and writes the draft with an LLM (OpenAI by default) | `.env` at the repo root |
| `credentials.json` | Gmail OAuth client, type *Desktop app*, with the Gmail API enabled in Google Cloud | the cookbook folder, next to `personalized_email.py` |
| `token.json` | Written on the first run, after you consent in the browser. Scope: `gmail.readonly` | the cookbook folder, created for you |
| `GRANOLA_API_KEY` | Reads your meeting notes through Granola's public API. Create one in Granola's settings | `.env` at the repo root |
| `MY_NAME` (optional) | Your name as it appears in your email, so the draft speaks as you | `.env` at the repo root |
| `cognee[gmail]` | The Google client libraries | `uv sync --extra gmail` |

No Granola? Run with `--no-granola` to skip that step.
`credentials.json` and `token.json` are git-ignored. Never commit or print them, or the keys.

## Try it on sample data

`setup.py` writes a sample mailbox and meetings to `sample/` (git-ignored), in the shape
Gmail and Granola give the scripts, dated relative to today. In a call twelve days ago,
Mira Lang promised Priya Shah pilot pricing by Friday and never sent it; yesterday Priya
wrote asking about it. The draft answers as Mira, whatever `MY_NAME` is. Only `LLM_API_KEY`
is needed: no Gmail or Granola account.

```bash
uv run python examples/cookbooks/personalized_email/personalized_email.py
```

With neither Gmail (`credentials.json`) nor `GRANOLA_API_KEY` set up, the script runs `setup.py` itself and uses the sample, as below. Pass `--sample` to
use it even when your own sources are set up.

```text
[ingest_granola] Remembered 2 sample meetings
[ingest_email] Remembered 2 sample inbox emails
[ingest_email] Remembered 3 sample sent emails
[draft] Answering: Pilot start and SSO (from Priya Shah <priya@northwind.example>)
[draft] Reply:

To: Priya Shah <priya@northwind.example>
Subject: Re: Pilot start and SSO

Hi Priya,

The pilot can start on the first of next month once the order form is signed.

Our SSO is SAML-only today; OIDC is on track and ships in the third week of next month.
Jonas can join a customer call to walk through the SSO setup if that would help.

I hadn't sent the pilot pricing yet. The pilot pricing is 3 seats at 400 EUR per month,
with the first month free.

Best,
M.
```

The facts come from the meetings, the admission that the pricing wasn't sent from the
sent mail, and the "Best, M." from how Mira writes.

## Steps

```
personalized_email/
├── README.md               this file
├── personalized_email.py   checks setup, then calls the scripts in order
├── setup.py                writes the sample, for --sample
├── sample/                 written by setup.py, git-ignored
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
| 1 | `scripts/ingest_granola.py [--days N] [--sample]` | Remembers Granola meeting notes from the last 30 days by default (node set `meetings`) | cognee dataset |
| 2 | `scripts/ingest_email.py [--emails N] [--sample]` | Remembers the newest 50 inbox and 50 sent emails by default, through cognee's Gmail connector `gmail_source` (node sets `inbox`, `sent_mail`) | cognee dataset |
| 3 | `scripts/draft.py [--sample]` | Drafts a reply to the newest inbox email: facts from a `HYBRID_COMPLETION` recall over the graph, tone from a `CHUNKS` recall over `sent_mail` | nothing |

All scripts use the cognee dataset `personalized_email`, named once in each script.

## Run it

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

## Clean up

```bash
uv run cognee-cli forget --dataset personalized_email
```
