# Import your coding-agent sessions

One memory of everything your coding agents did.

Claude Code and Codex keep every session on your machine, but each agent only sees its own,
and only until the session ends. This cookbook finds the agents installed on your machine,
reads their history (each prompt you typed and
the agent's answer to it), remembers it in one cognee dataset (`imported_agent_sessions`), and
answers questions over it: over every agent's history at once, and over each agent's alone.

cognee's databases are local files on your machine. The LLM is the one your key belongs to.

Agents run this cookbook through the `import-agent-sessions` skill,
[`.agents/skills/import-agent-sessions/SKILL.md`](../../../.agents/skills/import-agent-sessions/SKILL.md).

## Try it on sample sessions

`setup.py` writes four sample sessions to `sample/` (git-ignored): two from Claude Code and two
from Codex, two weeks of work on a made-up web shop, in the same file formats the agents write,
tool calls and injected instructions included. `--demo` imports both agents' samples, with any
one key, and asks the default question:

```bash
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --demo
```

```text
[setup] Demo: sample Claude Code and Codex sessions; cognee runs on anthropic claude-haiku-5-5 (ANTHROPIC_API_KEY).
[setup] Wrote the sample Claude Code and Codex sessions from setup.py.
[clear] Forgot the dataset imported_agent_sessions_demo
[read_sessions] Claude Code: 2 sessions, 4 turns
[read_sessions] Codex: 2 sessions, 2 turns
[ingest_sessions] Remembered 2 claude sessions as imported_agent_sessions_demo:claude_code
[ingest_sessions] Remembered 2 codex sessions as imported_agent_sessions_demo:codex
[ask] you> What is the topic of my data?
[ask] all sessions> Your sessions cover the acme-shop project, an online store. The work includes an order history page under /account/orders with a new orders(user_id, created_at) index (migration 0042), a fix for stock overselling at checkout (products.stock), token-bucket rate limiting on the public API, and a fix for double charges on checkout retry in src/checkout/.
[ask] Claude Code only> Your data covers two Claude Code sessions in the acme-shop project. One added an order history page at /account/orders ...
[ask] Codex only> Your data covers two Codex sessions on the acme-shop project. One fixed a checkout stock check that let two orders buy the last item ...
```

The demo has a dataset of its own (`imported_agent_sessions_demo`), forgotten at the start of
every demo run, so it never mixes with your imported sessions. `--ask` asks something else.
It does not look at your agents or your history.

## What it finds

Every run, `--check` included, starts by listing the agents on this machine:

```text
[find_agents] Claude Code (command line, VS Code extension): 645 sessions (605 terminal, 40 Agent SDK)
[find_agents] Codex (command line, VS Code extension): 467 sessions (391 IDE extension, 70 codex exec, 6 terminal)
[find_agents] Not imported: Cursor's own agent keeps its chats in Cursor's database, ...
```

How each agent is installed comes from the `claude` / `codex` command on `PATH`, the macOS
app, and the VS Code and Cursor extensions. Where each session was started comes from the
session file itself (`entrypoint` in Claude Code, `originator` in Codex). Every session in
`~/.claude/projects` and `~/.codex/sessions` is imported, wherever it was started; Codex's IDE
extension, for example, saves to the same folder as its command line.

Some installed apps keep chats this cookbook cannot read, and the run names them:

| App | Why it is not imported |
|---|---|
| Claude desktop app (its chats, not Claude Code) | Chats are stored in your Claude account. Settings > Privacy > Export data gives you `conversations.json` |
| Cursor (its own agent) | Chats stay in Cursor's database. Claude Code or Codex running inside Cursor are imported |
| ChatGPT app | Chats are stored in your OpenAI account |

## What it needs

At least one LLM key in `.env` at the repo root. Each key imports one agent's history, and the
first one found also runs cognee:

| Key | Imports | cognee runs on |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Code sessions, from `~/.claude/projects/*/*.jsonl` | `claude-haiku-5-5`, with local fastembed embeddings (Anthropic has no embedding API) |
| `OPENAI_API_KEY` | Codex sessions, from `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | OpenAI, cognee's default |
| both | both, into the same dataset | OpenAI |

An `LLM_API_KEY` that is already set runs cognee as `.env` configures it, and counts as the key
of its `LLM_PROVIDER` (`openai` by default, so it imports Codex history). `CLAUDE_CONFIG_DIR`
and `CODEX_HOME` move the history folders, as they do for the agents themselves.

## Run it

From the repo root:

```bash
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --check
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py
```

After listing your agents it asks which agents' sessions to ingest; Enter takes them all:

```text
[setup] cognee runs on openai (OPENAI_API_KEY).
[choose] Which agent sessions should be ingested into cognee?
[choose]   1) Claude Code: 645 sessions
[choose]   2) Codex: 467 sessions
[choose] Numbers separated by commas, or Enter for all:
[setup] Importing Claude Code and Codex history.
[read_sessions] Claude Code: 20 sessions, 164 turns
[read_sessions] Codex: 20 sessions, 31 turns
[ingest_sessions] Remembered 20 claude sessions as imported_agent_sessions:claude_code
[ingest_sessions] Remembered 20 codex sessions as imported_agent_sessions:codex
[ask] you> What is the topic of my data?
[ask] all sessions> Your sessions are mainly about ...
[ask] Claude Code only> ...
[ask] Codex only> ...
```

Only agents with sessions and a key are offered. With one agent chosen, only the
`all sessions` answer is printed. `--agents claude,codex` answers the question up front; without
a terminal to ask in (when an agent runs the cookbook) it is required, and a run without it
stops with `[setup] CHOOSE:` and exit 2.

Options:

| Option | Does |
|---|---|
| `--demo` | Imports the sample sessions from `setup.py`, both agents, with any one key. Combine with `--check` to only check the key |
| `--check` | Lists the agents, their sessions and the ones you can choose, and reports what is missing. Imports nothing |
| `--agents claude,codex` | Imports these agents without asking. Each needs its key. An agent with sessions but no key prints a `Skipped:` line naming the key |
| `--limit N` | Imports the newest `N` sessions per agent (default 20). Every session costs LLM calls, so start small |
| `--project TEXT` | Only sessions whose working folder contains `TEXT`, like `--project cognee` |
| `--since YYYY-MM-DD` | Only sessions started on or after that date |
| `--ask "..."` | Asks this instead of "What is the topic of my data?" |
| `--clear` | Forgets the dataset first, so only this run's sessions are in it |

Running it again adds new sessions and skips the ones whose content did not change. Exit codes:
0 done, 2 setup missing (each `[setup] MISSING:` line names one fix), 1 a step failed.

## Steps

```
import_agent_sessions/
├── README.md                   this file
├── import_agent_sessions.py    finds your agents, asks which to ingest, then calls the scripts in order
├── setup.py                    writes the sample sessions, for --demo
├── sample/                     written by setup.py, git-ignored
├── sessions/                   written by read_sessions.py, git-ignored
└── scripts/
    ├── find_agents.py
    ├── clear.py
    ├── read_sessions.py
    ├── ingest_sessions.py
    └── ask.py
```

| # | Command (`uv run python examples/cookbooks/import_agent_sessions/...`) | Does | Writes |
|---|---|---|---|
| 0 | `scripts/find_agents.py` | Lists the installed agents, their sessions by where they were started, and the apps it leaves out. Reads only the first lines of each session file | nothing |
| 1 | `scripts/clear.py` (or `--clear`) | Forgets the dataset `imported_agent_sessions`. Your files in `~/.claude` and `~/.codex` are never touched | cognee dataset |
| 2 | `scripts/read_sessions.py [--agents ...] [--limit N]` | Reads each agent's history and writes one Markdown file per session: your prompts and the agent's final answers, without tool calls, system reminders or injected instructions. No LLM, no cognee | `sessions/<agent>/*.md` |
| 3 | `scripts/ingest_sessions.py [--agents ...]` | Remembers the files, tagged with a node set per agent (`imported_agent_sessions:claude_code`, `imported_agent_sessions:codex`) | cognee dataset |
| 4 | `scripts/ask.py "question" [--agents ...]` | A `HYBRID_COMPLETION` recall over every chosen agent's node set, then one per agent's node set. Each answer runs in a new session, so earlier answers never leak into it | nothing |

The node sets carry the dataset's name because with `ENABLE_BACKEND_ACCESS_CONTROL=false` all
datasets share one graph, and a dataset filter alone would let other memories into the answers.

Open a file in `sessions/` to see exactly what cognee is given before you import more.

## What is kept, and what is not

- **Kept:** each prompt you typed (up to 2,000 characters), the agent's last answer to it (up to
  3,000), and the session's folder, git branch, start time and title. A session stops at about
  30,000 characters.
- **Dropped:** tool calls and their output, the agent's thinking, subagent transcripts, and the
  blocks the harness adds to a prompt (system reminders, `AGENTS.md`, environment context).
- **Redacted:** text that looks like an API key or token (`sk-...`, `ghp_...`, `AKIA...`,
  `xox...-`, `Bearer ...`) becomes `[redacted]`. This catches the common formats, not every
  secret, so read a few files in `sessions/` before importing a lot.
- **Skipped:** a session whose file changed in the last two minutes. That is a session still
  running, such as the one you run this cookbook from, and it would mostly describe the import.

## Share it with your agents

The dataset is ordinary cognee memory, so any agent connected to the same cognee can recall
it. The [Claude Code](https://docs.cognee.ai/cognee-cloud/agent-integrations/claude-code) and
Codex plugins record new sessions from the moment they are installed; this cookbook fills in
the history from before that. Point an agent at `imported_agent_sessions` to recall it.

## Clean up

```bash
uv run cognee-cli forget --dataset imported_agent_sessions
uv run cognee-cli forget --dataset imported_agent_sessions_demo
rm -r examples/cookbooks/import_agent_sessions/sessions examples/cookbooks/import_agent_sessions/sample
```
