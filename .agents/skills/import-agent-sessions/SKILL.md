---
name: import-agent-sessions
description: Import the user's own Claude Code and/or Codex session history (the prompts they typed and the agents' answers) into one cognee dataset, then answer questions over it, across every agent and per agent. Use when someone wants to load, import or backfill their previous coding-agent sessions into cognee, give several coding agents one shared memory of past work, or ask what their earlier sessions were about. Runs locally; reads ~/.claude and ~/.codex read-only, writes only to cognee's memory.
---

# Import coding-agent sessions

Runs the cookbook in `examples/cookbooks/import_agent_sessions/`. It first finds the coding
agents installed on this machine and the sessions each has kept. One LLM key
(`LLM_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) runs cognee and imports every agent's
history. The chosen agents go into the dataset `imported_agent_sessions`, and each question is
answered over all of them. Setup, the scripts and what each one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/import_agent_sessions/README.md).

Run every command from the repo root.

## Demo

When the user has no Claude Code or Codex history, or wants to see it work first, run the
demo instead of the steps below. It needs one key and imports sample sessions of both
agents (a made-up web shop), never the user's own:
`uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --demo`
(`--check` first to check the key; exit 2 with a `MISSING:` line means no key is set). Give
the user the `[ask]` answers and tell them they come from sample data. Pass `--ask "..."` for
their own question.

## Steps

1. Run `uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --check`.
   - Exit 2: each `[setup] MISSING:` line names one fix. Tell the user, using the README's
     "What it needs" table, and stop. Never ask the user to paste a key into the chat.
   - Exit 0: go on.
2. Tell the user what was found, from the `[find_agents]` lines: each agent, how it is
   installed, and how many sessions it has and where they were started. Name each
   `Not imported:` app (Claude desktop chats, Cursor's own agent) and why its chats can't be
   read.
3. Ask the user which agent sessions to ingest into cognee, with your question tool when you
   have one: one choice per `[setup] Can import:` line, several allowed. Skip asking only
   when the user already named the agents. Don't choose for them: without `--agents` the
   import stops with `[setup] CHOOSE:`, because the script cannot ask without a terminal.
4. Run the import with `--agents` set to the user's choice, like `--agents claude,codex`.
   Start with the default `--limit 20` unless the user asked for more, and pass the user's
   filters: `--project <text>` for one repo, `--since YYYY-MM-DD` for a period. Pass their
   question with `--ask "..."`; without one the run asks "What is the topic of my data?":
   `uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --agents <choice> --ask "..."`.
   It takes a few minutes for 20 sessions.
5. The answers are on the `[ask]` lines: `all sessions>` over everything, and with two agents
   one `<agent> only>` line each. Give them to the user.
6. Later questions don't need a new import:
   `uv run python examples/cookbooks/import_agent_sessions/scripts/ask.py "..." --agents claude,codex`
   (list only the agents the user chose).
7. If a step fails, the run exits 1 and the traceback's last line says why. Run that script
   from `examples/cookbooks/import_agent_sessions/scripts/` on its own to look closer.

## Rules

- Session history is the user's private data. Don't quote it beyond what answers the
  question, and don't print `.env`.
- Every imported session is sent to the provider of the key cognee runs on and costs LLM calls, so run the import only when the user asked for it,
  and raise `--limit` only when they want more history.
- Pass `--clear` only when the user asked to start over: it forgets everything the dataset
  holds. It never touches `~/.claude` or `~/.codex`.
- Redaction catches common key formats only. If the user is about to import a lot, suggest
  they look at a few files in `examples/cookbooks/import_agent_sessions/sessions/` first.

## Clean up

```bash
uv run cognee-cli forget --dataset imported_agent_sessions
uv run cognee-cli forget --dataset imported_agent_sessions_demo
rm -r examples/cookbooks/import_agent_sessions/sessions examples/cookbooks/import_agent_sessions/sample
```
