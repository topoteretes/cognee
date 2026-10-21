# Import your coding-agent sessions

One memory of everything your coding agents did.

Claude Code and Codex keep every session on your machine, but each agent only sees its own,
and only until the session ends. This cookbook finds the agents installed on your machine,
reads their history (each prompt you typed and
the agent's answer to it), remembers it in one cognee dataset (`imported_agent_sessions`), and
answers questions over it: over every agent's history at once, and over each agent's alone.

cognee's databases are local files on your machine. The LLM is the one your key belongs to.

Agents run this cookbook through the `multiagent-memory-setup` skill,
[`.agents/skills/multiagent-memory-setup/SKILL.md`](../../../.agents/skills/multiagent-memory-setup/SKILL.md).

## Try it on sample sessions

`setup.py` writes four sample sessions to `sample/` (git-ignored): two from Claude Code and two
from Codex, two weeks of work on a made-up web shop, in the same file formats the agents write,
tool calls and injected instructions included. Run without `--agents`, the cookbook runs this
demo: it imports both agents' samples and asks the default question:

```bash
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py
```

```text
[setup] Demo: sample Claude Code and Codex sessions; cognee runs on anthropic claude-haiku-5-5 (ANTHROPIC_API_KEY), local fastembed embeddings.
[setup] Wrote the sample Claude Code and Codex sessions from setup.py.
[clear] Forgot the dataset imported_agent_sessions_demo
[read_sessions] Claude Code: 2 sessions, 4 turns
[read_sessions] Codex: 2 sessions, 2 turns
[ingest_sessions] Remembered 2 claude sessions as imported_agent_sessions_demo:claude_code
[ingest_sessions] Remembered 2 codex sessions as imported_agent_sessions_demo:codex
[ask] you> What did I work on in these sessions?
[ask] all sessions> Your sessions cover the acme-shop project, an online store. The work includes an order history page under /account/orders with a new orders(user_id, created_at) index (migration 0042), a fix for stock overselling at checkout (products.stock), token-bucket rate limiting on the public API, and a fix for double charges on checkout retry in src/checkout/.
[ask] Claude Code only> Your data covers two Claude Code sessions in the acme-shop project. One added an order history page at /account/orders ...
[ask] Codex only> Your data covers two Codex sessions on the acme-shop project. One fixed a checkout stock check that let two orders buy the last item ...
[setup] Those answers come from sample sessions. To import your own, pass --agents claude, codex or both (--check lists what you have).
```

The demo has a dataset of its own (`imported_agent_sessions_demo`), forgotten at the start of
every demo run, so it never mixes with your imported sessions. `--ask` asks something else.
It does not look at your agents or your history. `--limit`, `--project` and `--since` choose
from your own sessions, so they need `--agents`; without it the run stops with `[setup] MISSING:`.

## What it finds

`--check` and every run with `--agents` start by listing the agents on this machine
and the project folders their sessions were started in:

```text
[find_agents] Claude Code (command line, VS Code extension): 645 sessions (605 terminal, 40 Agent SDK)
[find_agents] Codex (command line, VS Code extension): 467 sessions (391 IDE extension, 70 codex exec, 6 terminal)
[find_agents] Project: /Users/you/code/shop (Claude Code 240, Codex 210)
[find_agents] Project: /Users/you/code/client-app (Claude Code 130)
[find_agents] Not imported: Cursor's own agent keeps its chats in Cursor's database, ...
```

How each agent is installed comes from the `claude` / `codex` command on `PATH`, the macOS
app, and the VS Code and Cursor extensions. Where each session was started comes from the
session file itself (`entrypoint` in Claude Code, `originator` in Codex), and so does its project
folder (the session's working directory). Sessions from `~/.claude/projects` and
`~/.codex/sessions` are imported wherever they were started; Codex's IDE extension, for example,
saves to the same folder as its command line. `--project` keeps the import to the projects you
name, so sessions from other repos (a client's, say) are never sent to the LLM.

Some installed apps keep chats this cookbook cannot read, and the run names them:

| App | Why it is not imported |
|---|---|
| Claude desktop app (its chats, not Claude Code) | Chats are stored in your Claude account. Settings > Privacy > Export data gives you `conversations.json` |
| Cursor (its own agent) | Chats stay in Cursor's database. Claude Code or Codex running inside Cursor are imported |
| ChatGPT app | Chats are stored in your OpenAI account |

## What it needs

One LLM key in `.env` at the repo root. It runs cognee's LLM, and cognee imports every agent's
history with it, Claude Code's (`~/.claude/projects/*/*.jsonl`) and Codex's
(`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`) alike. Whichever key it is, every imported
session is sent to that provider.

| Key | cognee's LLM |
|---|---|
| `LLM_API_KEY` | what `.env` configures (`LLM_PROVIDER`, `LLM_MODEL`), used as it is |
| `OPENAI_API_KEY` | OpenAI, cognee's default |
| `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

The first one set in that order wins. Embeddings are always local fastembed
(`BAAI/bge-small-en-v1.5`, downloaded once, about 70 MB), whichever key it is, unless `.env`
sets `EMBEDDING_PROVIDER`. A dataset's vectors must all come from one model, so this lets you
switch keys between runs. If you do set `EMBEDDING_PROVIDER`, keep it for every run, or
`--clear` first. With `ENABLE_BACKEND_ACCESS_CONTROL=false` all datasets share one vector
store, so it also has to match what your other datasets were embedded with.
`scripts/configure.py` makes these choices, and every script that calls cognee uses it, so each
also runs alone. `CLAUDE_CONFIG_DIR` and `CODEX_HOME` move the history
folders, as they do for the agents themselves.

## Run it

From the repo root:

```bash
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --check
uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --agents claude,codex
```

`--check` lists your agents and the projects their sessions were started in, so you can pick.
`--agents` names the agents whose history to import, and only agents with sessions can be named:

```text
[setup] cognee runs on openai (OPENAI_API_KEY), local fastembed embeddings.
[setup] Importing Claude Code and Codex history.
[read_sessions] Claude Code: 20 sessions, 164 turns
[read_sessions] Codex: 20 sessions, 31 turns
[ingest_sessions] Remembered 20 claude sessions as imported_agent_sessions:claude_code
[ingest_sessions] Remembered 20 codex sessions as imported_agent_sessions:codex
[ask] you> What did I work on in these sessions?
[ask] all sessions> Your sessions are mainly about ...
[ask] Claude Code only> ...
[ask] Codex only> ...
```

With one agent named, only the `all sessions` answer is printed.

Options:

| Option | Does |
|---|---|
| `--agents claude,codex` | Imports these agents' history. Without it the run is the demo on the sample sessions from `setup.py` |
| `--check` | Lists the agents, their sessions and projects, and reports what is missing. Imports nothing |
| `--limit N` | Imports the newest `N` sessions per agent (default 20). Every session costs LLM calls, so start small |
| `--project PATH` | Only sessions started in this folder or inside it, like `--project ~/code/cognee` (not `~/code/cognee-docs`). Repeat it for several projects. The paths are the ones on the `[find_agents] Project:` lines |
| `--since YYYY-MM-DD` | Only sessions started on or after that date |
| `--ask "..."` | Asks this instead of "What did I work on in these sessions?" |
| `--clear` | Forgets the dataset first, so only this run's sessions are in it |

Running it again adds new sessions and skips the ones whose content did not change. A session
that went on since the last import (you resumed it) is imported again, and its earlier copy is
forgotten first, so the dataset never holds two versions of one session. Exit codes:
0 done, 2 setup missing (each `[setup] MISSING:` line names one fix), 1 a step failed.

## Steps

```
import_agent_sessions/
├── README.md                   this file
├── import_agent_sessions.py    the demo, or finds your agents and imports the ones you name
├── setup.py                    writes the sample sessions, for the demo
├── sample/                     written by setup.py, git-ignored
├── sessions/                   written by read_sessions.py, git-ignored
└── scripts/
    ├── configure.py            which LLM key cognee runs on, and its local embeddings
    ├── find_agents.py
    ├── clear.py
    ├── read_sessions.py
    ├── ingest_sessions.py
    └── ask.py
```

| # | Command (`uv run python examples/cookbooks/import_agent_sessions/...`) | Does | Writes |
|---|---|---|---|
| 0 | `scripts/find_agents.py` | Lists the installed agents, their sessions by where they were started and by project folder, and the apps it leaves out. Reads only the first lines of each session file | nothing |
| 1 | `scripts/clear.py` (or `--clear`) | Forgets the dataset `imported_agent_sessions`. Your files in `~/.claude` and `~/.codex` are never touched | cognee dataset |
| 2 | `scripts/read_sessions.py [--agents ...] [--limit N] [--project PATH]` | Reads each agent's history and writes one Markdown file per session: your prompts and the agent's final answers, without tool calls, system reminders or injected instructions. No LLM, no cognee | `sessions/<agent>/*.md` |
| 3 | `scripts/ingest_sessions.py [--agents ...]` | Remembers the files, tagged with a node set per agent (`imported_agent_sessions:claude_code`, `imported_agent_sessions:codex`) | cognee dataset |
| 4 | `scripts/ask.py "question" [--agents ...]` | A `HYBRID_COMPLETION` recall over every chosen agent's node set, then one per agent's node set. Each answer runs in a new session, so earlier answers never leak into it | nothing |

The node sets carry the dataset's name because with `ENABLE_BACKEND_ACCESS_CONTROL=false` all
datasets share one graph, and a dataset filter alone would let other memories into the answers.

Open a file in `sessions/` to see exactly what cognee is given before you import more.

## What is kept, and what is not

- **Kept, in full:** each prompt you typed, the agent's last answer to it, and the session's
  folder, git branch, start time and title. Long sessions are not cut; cognee chunks them.
- **Dropped:** tool calls and their output, the agent's thinking, subagent transcripts, the
  blocks the harness adds to a prompt (system reminders, `AGENTS.md`, environment context, the
  IDE's open tabs), the summary `/compact` writes, interrupt markers, and `!` shell commands
  run in Claude Code with their output.
- **Redacted:** in prompts, answers and titles, these become `[redacted]`:
  - keys and tokens by their format: OpenAI and Anthropic (`sk-...`), GitHub (`ghp_...`,
    `github_pat_...`), GitLab (`glpat-...`), AWS key ids (`AKIA...`), Slack (`xox...-`), Stripe
    (`sk_live_...`, `rk_...`, `whsec_...`), Google (`AIza...`), Hugging Face (`hf_...`), npm,
    JWTs, `Bearer ...` and PEM private keys
  - the password in a URL, like `postgresql://user:[redacted]@host`
  - a value of 8 characters or more assigned to a name that says it is secret (`DB_PASSWORD=...`,
    `"api_key": "..."`, `auth_token: ...`), whatever its format. Code that assigns such a name
    (`token = request.headers[...]`) is redacted too.

  This catches the common shapes, not every secret, so read a few files in `sessions/` before
  importing a lot.
- **Skipped:** a session whose file changed in the last two minutes. That is a session still
  running, such as the one you run this cookbook from, and it would mostly describe the import.

## Share it with your agents

The dataset is ordinary cognee memory in this repo's databases (`.venv` by default, or the
`SYSTEM_ROOT_DIRECTORY` and `DATA_ROOT_DIRECTORY` your `.env` sets). Every agent can ask it
through `scripts/ask.py`, which is what the skill does.

The [Claude Code](https://docs.cognee.ai/cognee-cloud/agent-integrations/claude-code) and Codex
plugins record new sessions from the moment they are installed, but they do not see this
import yet:

- They run their own cognee server (`http://localhost:8011`, data in `~/.cognee/`), a different
  database from this cookbook's.
- They recall from their active dataset, `agent_sessions` by default, not
  `imported_agent_sessions`.
- With `COGNEE_PROJECT_NODE_SET` set, they recall only their project node sets, which
  `imported_agent_sessions:claude_code` and `imported_agent_sessions:codex` are not.
- The import is stored as documents, not as session entries, so the plugins' recap, standup and
  digest never list these sessions.

## Clean up

```bash
uv run cognee-cli forget --dataset imported_agent_sessions
uv run cognee-cli forget --dataset imported_agent_sessions_demo
rm -r examples/cookbooks/import_agent_sessions/sessions examples/cookbooks/import_agent_sessions/sample
```
