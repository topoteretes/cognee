# Google Workspace memory

One memory of your Google Calendar, Gmail and Drive that you can ask questions across.

cognee remembers your primary calendar, your newest inbox emails and one Drive folder in one
dataset (`google_workspace_memory`). Because the same people, clients and orders appear in
all three, a single question can connect a meeting to the emails and sheets about it. By
default it prepares your coming client meetings: which orders are late or at risk, why, and
what to tell each client.

Agents run this cookbook through the `google-workspace-memory` skill,
[`.agents/skills/google-workspace-memory/SKILL.md`](../../../.agents/skills/google-workspace-memory/SKILL.md).

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and answers with an LLM (OpenAI by default) | `.env` at the repo root |
| `credentials.json` | Google OAuth client, type *Desktop app*, with the Calendar, Gmail and Drive APIs enabled in Google Cloud | the cookbook folder, next to `google_workspace_memory.py` |
| `calendar_token.json`, `gmail_token.json`, `drive_token.json` | Written on the first run, after you consent in the browser, one per app. Scopes: `calendar.readonly`, `gmail.readonly`, `drive.readonly` | the cookbook folder, created for you |
| `GOOGLE_DRIVE_FOLDER_ID` | The Drive folder to remember: the ID at the end of its URL | `.env` at the repo root |
| `cognee[gmail,google-drive,google-calendar]` | The Google client libraries; `--check` reports them missing | `uv sync --extra gmail --extra google-drive --extra google-calendar` |

To create `credentials.json`: in the Google Cloud Console, enable the Google Calendar, Gmail
and Google Drive APIs. In Google Auth Platform, add your account as a test user (Audience),
add the three read-only scopes (Data Access), then create a *Desktop app* client (Clients)
and download its JSON as `credentials.json`.

No Drive folder? Run with `--no-drive`. `--no-calendar` and `--no-gmail` skip the other
apps. `credentials.json` and the token files are git-ignored. Never commit or print them,
or the keys.

## Try it on sample data

`setup.py` writes a sample workspace to `sample/` (git-ignored), in the shape the three
Google connectors give the scripts, dated relative to today. Ana Novak runs Birchwood
Ceramics, which makes tableware for restaurants. Her orders sheet, her inbox and her
calendar show that she meets Harbor Café in two days and Olive & Fig in four, that a
courier delayed Harbor Café's plates past their menu launch, and that a back-ordered glaze
puts Olive & Fig's pasta bowls at risk. Only `LLM_API_KEY` is needed: no Google account.

```bash
uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py
```

Without `credentials.json`, the script runs `setup.py` itself and uses the sample, as below.
Pass `--sample` to use it even when your own Google account is set up. The sample lives in
its own dataset, `google_workspace_memory_sample`, so it never mixes with your own data. A
sample run first forgets that dataset, so copies from an earlier run (the sample is dated
relative to today, so it changes daily) never mix with this one. `--no-clear` keeps it.

```text
[clear] Forgot the dataset google_workspace_memory_sample
[ingest_calendar] Remembered 4 sample calendar events
[ingest_gmail] Remembered 4 sample emails
[ingest_drive] Remembered 1 sample Drive file
[ask] Which client meetings do I have in the next seven days? For each of those clients, which orders are late or at risk of missing the promised delivery date, why, and what should I tell them in the meeting?
[ask] Answer:

Next 7 days client meetings:
- Quarterly review with Harbor Café — 10:00, in two days (Location: Harbor Café)
- Order check-in with Olive & Fig — 14:00, in four days (Google Meet)

1) Harbor Café
- At-risk order: BC-1042 — Dinner plates 27 cm (80). Promised for tomorrow; status
  "Shipped", but Swift Freight reports a pallet damaged at the Riverside depot, being
  repacked and now delivered in three days.
- What to tell them: apologise, explain the courier delay and the new delivery date, which
  misses the evening-before setup for their menu launch, and offer to expedite.

2) Olive & Fig
- At-risk order: BC-1043 — Pasta bowls (sage glaze). The supplier back-ordered the sage
  glaze until after the promised date.
- BC-1044 — Serving platters (white): no reported delay.
- What to tell them: the platters can still make the date, but the bowls can't. Sofia wants
  both to arrive together, so ask whether to ship the platters first or hold both.
```

The meetings come from the calendar, the promised dates and statuses from the orders sheet,
and the reasons from the courier's and the supplier's emails.

## Steps

```
google_workspace_memory/
├── README.md                    this file
├── google_workspace_memory.py   checks setup, then calls the scripts in order
├── setup.py                     writes the sample, for --sample
├── sample/                      written by setup.py, git-ignored
├── credentials.json             yours, git-ignored
├── *_token.json                 yours, git-ignored, written on the first run of each app
└── scripts/
    ├── clear.py
    ├── ingest_calendar.py
    ├── ingest_gmail.py
    ├── ingest_drive.py
    └── ask.py
```

`google_workspace_memory.py` imports each script and calls its function in one process. Each
script also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/google_workspace_memory/...`) | Does | Writes |
|---|---|---|---|
| 0 | `google_workspace_memory.py --check` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/clear.py [--sample]` (or `--clear`) | Forgets the dataset `google_workspace_memory` (with `--sample`, `google_workspace_memory_sample`). Runs first with `--clear`, which is on by default for a sample run (`--no-clear` turns it off) | cognee dataset |
| 2 | `scripts/ingest_calendar.py [--sample]` | Remembers every event of your primary calendar, past and future, through cognee's Google Calendar connector `google_calendar_source` (node set `calendar`) | cognee dataset |
| 3 | `scripts/ingest_gmail.py [--emails N] [--sample]` | Remembers the newest 50 inbox emails by default (`--emails N`, at least 1), through cognee's Gmail connector `gmail_source` (node set `email`). Emails that leave the newest N are forgotten on the next run | cognee dataset |
| 4 | `scripts/ingest_drive.py [--sample]` | Remembers the Docs, Sheets (first sheet, as CSV), PDFs and text files in `GOOGLE_DRIVE_FOLDER_ID` and its subfolders (PDFs and text files over 25 MB are skipped; `GOOGLE_DRIVE_INCLUDE_SUBFOLDERS` and `GOOGLE_DRIVE_MAX_FILE_SIZE_MB` in `.env` change these defaults), through cognee's Google Drive connector `google_drive_source` (node set `drive`) | cognee dataset |
| 5 | `scripts/ask.py [--ask "question"] [--sample]` | Answers one question across all three with a `HYBRID_COMPLETION` recall. Today's date goes into the question; the default prepares your client meetings of the next seven days | nothing |

All scripts use the cognee dataset `google_workspace_memory`, or `google_workspace_memory_sample`
with `--sample`, named once in each script.

## Run it

From the repo root:

```bash
uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --check
uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py
uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --ask "What did Leo ask me about last week?"
```

The output looks like this (an illustration; yours comes from your own account):

```text
[ingest_calendar] Remembering your primary calendar. The first run reads every event and can take ten minutes or more; later runs fetch only changes.
[ingest_calendar] Remembered your primary calendar: 262 new or changed events, 0 deleted
[ingest_gmail] Remembering your newest 50 inbox emails…
[ingest_gmail] Remembered your newest inbox emails: 50 read
[ingest_drive] Remembering your Drive folder. The first run reads every file in it…
[ingest_drive] Remembered your Drive folder: 3 new or changed files checked
[ask] Which client meetings do I have in the next seven days? ...
[ask] Answer:

...
```

The first run opens the browser three times, once per app, to consent. It reads your whole
primary calendar and every supported file in the folder and its subfolders, and cognifies
each item with your LLM: a calendar with a few hundred events takes about ten minutes. Later
runs are quick: Calendar and Drive fetch only what changed since the last run, and deleted
events and files are forgotten. Gmail reads the newest emails again each run. cognee skips
the ones it already holds, unless their labels changed (read, starred), and forgets those
no longer among the newest, so memory holds your inbox as it is now. Add `--clear` to
forget the dataset `google_workspace_memory` first and start over.

## Clean up

```bash
uv run cognee-cli forget --dataset google_workspace_memory
uv run cognee-cli forget --dataset google_workspace_memory_sample
```
