# Day-one seeding

On a fresh install, cognee can immediately ingest the knowledge that already
exists around your workspace — so the **first `recall` has something to
return** instead of an empty result:

| Category | node_set | What is picked up |
| --- | --- | --- |
| Agent memory | `agent_memory` | `MEMORY.md`, `SOUL.md`, `AGENTS.md`, `CLAUDE.md`, `USER.md`, `IDENTITY.md`, `TOOLS.md` at the workspace root; `memory/*.md`; `.claude/CLAUDE.md`; Claude Code auto-memory for this workspace |
| Workspace docs | `workspace_docs` | `README.*` at the workspace root |
| Session logs | `session_logs` | Recent sessions for this workspace from known coding agents (newest 3 per agent by default, ≤5 MB each) — see the adapter table below |
| Codebase | `codebase` | The workspace itself, when it is a code project — resolved through the repo-manifest ingestion path |

Everything lands in one dataset (default `workspace`), and a default
`recall()` searches every dataset you own, so no dataset argument is needed.

Seeding runs in stages ordered by size — memory files and the README first —
so a recall issued moments after install already hits stage-1 knowledge.
`add()` is LLM-free; if no LLM key is configured yet, the data is still
ingested and `cognify` is deferred to the next run with a key.

## Automatic (MCP server)

The cognee MCP server seeds automatically on startup when all of these hold:

- local mode (no `--api-url` / cloud connection),
- the user owns **zero datasets** (a genuinely fresh install — it never
  re-seeds behind your back),
- `COGNEE_AUTO_SEED` is not set to `false`.

Seeding runs in the background; server startup never blocks on it.

## Manual (CLI)

```bash
cognee-cli seed --dry-run     # print what would be ingested, ingest nothing
cognee-cli seed               # seed the current workspace
cognee-cli seed --no-code --no-session-logs
cognee-cli seed --force       # re-seed an existing seed dataset
cognee-cli seed -w /path/to/workspace -d my_dataset
```

## SDK

```python
import cognee

result = await cognee.seed()          # auto-detects the workspace root
print(result.summary())
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `COGNEE_AUTO_SEED` | `true` | MCP-server auto-seed on fresh installs |
| `COGNEE_SEED_CODEBASE` | `true` | Include the codebase stage |
| `COGNEE_SEED_SESSION_LOGS` | `true` | Include session transcripts |
| `COGNEE_SEED_MAX_SESSION_LOGS` | `3` | Newest N transcripts to ingest |
| `COGNEE_SEED_MAX_SESSION_LOG_BYTES` | `5242880` | Per-transcript size cap |
| `COGNEE_SEED_MAX_FILE_BYTES` | `10485760` | Per-file size cap for memory/docs |

There is intentionally no free-form path/glob override: cognee loads `.env`
from the working directory, so an env-driven glob would let a hostile repo
point the seeder at arbitrary files. Support for more agents' transcript
locations lands as explicit discovery adapters.

## Session-log adapters

Each adapter is scoped to *this* workspace and capped to the newest
`COGNEE_SEED_MAX_SESSION_LOGS` entries:

| Agent | Location | Workspace mapping |
| --- | --- | --- |
| Claude Code | `~/.claude/projects/<slug>/*.jsonl` | slug = absolute path, non-alphanumerics → `-` |
| Codex CLI | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | first-line session meta `payload.cwd` (top-level `cwd` in older layouts); newest 500 rollouts scanned |
| Gemini CLI | `~/.gemini/tmp/<hash>/logs.json` + `chats/*.json` | hash = SHA-256 hex of the absolute project root |
| pi | `~/.pi/agent/sessions/--<slug>--/*.jsonl` | slug = path with `/` → `-`, wrapped in `--` (observed, not officially documented) |
| Aider | `.aider.chat.history.md` | lives at the workspace root (explicitly allowlisted dotfile) |

Not yet supported (session stores are binary/SQLite rather than files):
Cursor CLI (`~/.cursor/chats/**/store.db`), opencode ≥1.2
(`~/.local/share/opencode/opencode.db`), Amp (server-synced
`~/.local/share/amp/threads/*.json`, mapping undocumented).

## Safety

- Hidden files are never picked up, with one explicit allowlist entry
  (`.claude/CLAUDE.md`) — `.env` and friends can never enter the seed.
- Sources outside cognee's allowed local-file roots (session transcripts,
  Claude Code auto-memory) are staged into
  `{data_root}/seed_staging/<category>/` before ingestion, keeping the path
  allowlist intact for every other caller.
- Session transcripts can contain secrets echoed by tools; cap sizes are
  enforced and the category is off-switchable.
- Re-running is safe: seeding refuses to touch an existing seed dataset
  without `--force`, and content-hash dedup makes `--force` cheap on
  unchanged files.
