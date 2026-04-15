# Community Support Bot

Multi-scope memory Discord support bot built on Cognee.

## What this is

A Discord bot that ingests project knowledge from multiple sources (docs, GitHub issues, code-agent Q&A, past Discord threads) into a Cognee knowledge graph, then answers community questions by pulling context from three separate memory scopes:

- **Org memory** — shared project knowledge
- **User memory** — per-Discord-user history and context
- **Agent memory** — patterns the bot learned from its own work

See `/Users/veljko/.claude/plans/logical-crunching-patterson.md` for the full plan.

## Layout

```
community-bot/
├── ingest/           # offline sync workers
│   ├── docs.py       # MDX from docs repo  → org dataset
│   ├── github_issues.py  # GitHub issues  → org dataset
│   ├── code_qa.py    # Claude-Code-produced Q&A → org dataset  (Day 2)
│   └── discord_history.py  # past channel history → org dataset  (Day 4)
├── bot/              # discord.py bot (Day 3)
├── dashboard/        # Flask review UI      (Day 4)
├── demo/             # seed data for demos   (Day 5)
├── config.py         # dataset names, env lookups
├── smoke_test.py     # Day 1 verification
└── .env.example
```

## Setup

```bash
# From repo root, activate the existing cognee venv
source .venv/bin/activate

# Community-bot extra deps
pip install -r community-bot/requirements.txt

# Fill in secrets
cp community-bot/.env.example community-bot/.env
# edit community-bot/.env
```

## Day 1: populate org dataset from docs + issues

```bash
python -m community-bot.ingest.docs
python -m community-bot.ingest.github_issues
python community-bot/smoke_test.py
```
