# 🤖 Support-Triage Bot — Powered by Cognee Memory

A bot that watches a support channel, **recalls similar past issues + resolutions** from cognee memory, and **suggests answers with citations** to prior threads. Implements [issue #3615](https://github.com/topoteretes/cognee/issues/3615).

## ✨ Features

- **Intelligent Triage**: New support questions are matched against a memory of resolved issues using `cognee.recall()`
- **Cited Suggestions**: Responses include numbered citations linking back to source threads
- **Human-in-the-Loop**: Suggestions are delivered as ephemeral messages — agents review before publishing
- **Thread Ingestion**: Resolved threads are stored in memory via `cognee.remember()` for future recall
- **Forget + Opt-Out**: Users can remove specific threads from memory or opt out of future ingestion
- **Platform Agnostic**: Abstract channel adapter supports Slack, CLI, and future platforms

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- Python 3.10+
- cognee installed (you're in the cognee repo!)
- `.env` with `LLM_API_KEY` configured

### 1. Install Dependencies

```bash
cd cognee
uv sync   # or pip install -e .
```

### 2. Run in CLI Mode (No External Tokens)

```bash
cd examples/pocs/support_triage_bot
python run_bot.py
```

### 3. Run with Pre-Seeded Demo Threads

```bash
python run_bot.py --seed
```

This seeds 3 resolved support threads into memory, then drops you into an interactive REPL:

```
🌱 Seeding demo threads into cognee memory…
  ✅ Seeded T001: Auth timeout after token refresh…
  ✅ Seeded T002: Session expiry on mobile app…
  ✅ Seeded T003: Database connection pool exhaustion…

support> auth timeout after token refresh
🔍 Searching for similar past issues…

💡 Similar past issues found:

[1] Auth timeout after token refresh: Bumped token TTL from 1h to 24h
    Thread: T001 | Score: 0.95
    → https://support.example.com/threads/T001

[2] Session expiry on mobile app: Same root cause — mobile SDK used shorter TTL
    Thread: T002 | Score: 0.85
    → https://support.example.com/threads/T002

Suggested fix: Based on past resolutions, this may be related to: …

React ✅ to save this thread | Type `!forget` to remove from memory
```

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `<any text>` | Triage: find similar past issues |
| `!resolve <id> <msg1\|msg2\|msg3>` | Ingest a resolved thread (pipe-separated messages) |
| `!forget <id>` | Remove a thread from memory |
| `!optout` | Opt out of future ingestion |
| `!status` | Show stored thread ↔ data_id mappings |
| `!quit` | Stop the bot |

## 🔌 Slack Mode (Optional)

### Setup

1. Create a [Slack App](https://api.slack.com/apps) with these Bot Token Scopes:
   - `chat:write`
   - `reactions:read`
   - `channels:history`
   - `channels:read`

2. Enable Event Subscriptions for:
   - `message.channels`
   - `reaction_added`

3. Set environment variables:
   ```
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   SLACK_SIGNING_SECRET=...
   SUPPORT_CHANNEL_ID=C...
   ```

4. Run:
   ```bash
   python run_bot.py --slack
   ```

## ⚙️ Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPPORT_BOT_DATASET` | `support_threads` | Cognee dataset name |
| `SUPPORT_BOT_MEMORY_SCOPE` | `channel` | `channel` or `workspace` |
| `SUPPORT_BOT_TOP_K` | `5` | Max recall results |
| `SUPPORT_BOT_MIN_RELEVANCE` | `0.0` | Minimum relevance threshold |
| `SUPPORT_BOT_EPHEMERAL` | `true` | Ephemeral replies (Slack) |
| `SUPPORT_BOT_AUTO_INGEST` | `true` | Auto-ingest on resolve |
| `RESOLVE_EMOJI` | `white_check_mark` | Emoji that triggers ingestion |

## 🧪 Running Tests

All tests are **fully deterministic** — no real LLM keys, no platform tokens, no network calls:

```bash
cd cognee
uv run pytest cognee/tests/unit/pocs/support_triage_bot/ -v
```

## 🏗️ Architecture

```
Support Channel (Slack / CLI / ...)
        │
        ▼
 Channel Adapter (abstract interface)
        │
        ▼
 Event Router
        │
        ├── New message ──→ TriageHandler ──→ cognee.recall() ──→ Citations
        ├── ✅ Reaction  ──→ IngestHandler ──→ cognee.remember() ──→ Memory
        ├── !forget      ──→ ForgetHandler ──→ cognee.forget()
        └── !optout      ──→ OptOutHandler ──→ Blocklist
```

### Memory Boundary

| Support concept | Cognee concept | Isolation |
|----------------|---------------|-----------|
| `workspace_id` | `dataset_name` | Cross-channel search via config |
| `channel_id` | `session_id` | Default: channel-scoped |
| `thread_ts` | `data_id` (UUID) | Adapter maintains persistent mapping |

## 📁 File Structure

```
examples/pocs/support_triage_bot/
├── README.md               # This file
├── run_bot.py               # Entry point (CLI or Slack)
├── config.py                # Configuration dataclass
├── models.py                # SupportThread, Citation, TriageResult
├── memory_adapter.py        # Thin wrapper around cognee APIs
├── handlers.py              # Triage, Ingest, Forget, OptOut
├── citation_builder.py      # Format citations for display
└── channel_adapters/
    ├── __init__.py
    ├── base.py              # Abstract ChannelAdapter
    ├── cli_adapter.py       # CLI for development/testing
    └── slack_adapter.py     # Slack implementation

cognee/tests/unit/pocs/support_triage_bot/
├── __init__.py
├── conftest.py              # Fixtures + mock cognee
├── test_models.py           # Model validation
├── test_memory_adapter.py   # Memory operations + ID mapping
├── test_handlers.py         # Event handler logic
└── test_citation_builder.py # Citation formatting
```

## 🔗 Related Issues

- [#3615](https://github.com/topoteretes/cognee/issues/3615) — This issue
- [#3608](https://github.com/topoteretes/cognee/issues/3608) — Shared chat-memory adapter
- [#3604](https://github.com/topoteretes/cognee/issues/3604) — Citable results
- [#3601](https://github.com/topoteretes/cognee/issues/3601) — Mocked tests
- [#3605](https://github.com/topoteretes/cognee/issues/3605) — Easy onboarding
