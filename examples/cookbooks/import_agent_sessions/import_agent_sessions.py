"""Import your Claude Code and Codex session history into cognee, then ask about it.

    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py
    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --check
    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py \
        --agents claude,codex --limit 50 --project ~/code/cognee --ask "Which tickets did I work on?"

Without --agents it runs the demo: it imports the sample Claude Code and Codex sessions from
setup.py into a dataset of its own, so it works with no history and never reads yours. With
--agents it lists the coding agents installed on this machine and the sessions each has kept,
then imports the named agents' history. One LLM key runs cognee and imports every agent's
history: an OpenAI or an Anthropic key, or the LLM_API_KEY your .env configures; embeddings are
local (scripts/configure.py). Everything goes into one dataset, so every question is answered
over all the agents you chose. --check lists your agents too, and imports nothing. Each script
in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script failed (its
traceback says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.ask import ask
from scripts.clear import clear
from scripts.configure import NO_KEY, configure_cognee, has_key
from scripts.find_agents import Agent, describe, find_agents, not_read, projects
from scripts.ingest_sessions import DATASET, ingest_sessions
from scripts.read_sessions import AGENTS, OUT, read_sessions, write_sessions
from setup import SAMPLE, write_sample

DEFAULT_QUESTION = "What did I work on in these sessions?"
DEMO_DATASET = "imported_agent_sessions_demo"  # never mixes with your own imported sessions
DEFAULT_LIMIT = 20


def report(found: list[Agent]) -> None:
    """Print the agents on this machine, their project folders, and the apps left out."""
    for agent in found:
        print(f"[find_agents] {describe(agent)}")
    for line in projects(found):
        print(f"[find_agents] Project: {line}")
    for line in not_read():
        print(f"[find_agents] Not imported: {line}.")


def missing_setup(args: argparse.Namespace, found: list[Agent]) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    with_sessions = [agent.name for agent in found if agent.sessions]
    if not with_sessions:
        return ["No Claude Code or Codex sessions on this machine, so there is nothing to import."]
    if not has_key():
        return [NO_KEY]
    missing = []
    for project in args.project or []:
        if not Path(project).expanduser().is_absolute():
            missing.append(
                f"--project {project} is not a folder path; pass one from a"
                " [find_agents] Project: line, like ~/code/cognee."
            )
    for agent in args.agents:
        if agent not in AGENTS:
            missing.append(f"--agents names {agent}; use claude, codex or both.")
        elif agent not in with_sessions:
            missing.append(f"--agents names {agent}, but it has no sessions on this machine.")
    return missing


async def run(args: argparse.Namespace, dataset: str = DATASET, out=OUT) -> None:
    if args.clear:
        await clear(dataset)
    for agent in args.agents:
        sessions = read_sessions(agent, args.limit, args.project, args.since)
        write_sessions(sessions, agent, out)
        turns = sum(len(session.turns) for session in sessions)
        print(f"[read_sessions] {AGENTS[agent]}: {len(sessions)} sessions, {turns} turns")
    await ingest_sessions(args.agents, dataset, out)
    await ask(args.ask, args.agents, dataset)


def run_demo(args: argparse.Namespace) -> None:
    """Import the sample sessions from setup.py, both agents, into the demo's own dataset."""
    write_sample()
    print("[setup] Wrote the sample Claude Code and Codex sessions from setup.py.")
    os.environ["CLAUDE_CONFIG_DIR"] = str(SAMPLE / "claude")  # the readers look here instead
    os.environ["CODEX_HOME"] = str(SAMPLE / "codex")  # of ~/.claude and ~/.codex
    args.agents, args.clear = list(AGENTS), True  # the demo always starts from an empty dataset
    asyncio.run(run(args, DEMO_DATASET, SAMPLE / "rendered"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument(
        "--agents",
        help="claude, codex or both: import their history (default: the demo on sample sessions)",
    )
    parser.add_argument(
        "--limit", type=int, help=f"newest sessions per agent (default {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--project",
        action="append",
        help="only sessions in this folder or inside it; repeat for several projects",
    )
    parser.add_argument("--since", help="only sessions started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--clear", action="store_true", help="forget the dataset before importing")
    parser.add_argument("--ask", default=DEFAULT_QUESTION, help=f'default: "{DEFAULT_QUESTION}"')
    args = parser.parse_args()
    args.agents = args.agents.split(",") if args.agents else []

    if not args.agents:  # no agents named: the demo, on the sample sessions
        own_only = [
            flag
            for flag, value in (
                ("--limit", args.limit),
                ("--project", args.project),
                ("--since", args.since),
            )
            if value is not None
        ]
        if own_only:
            print(
                f"[setup] MISSING: {', '.join(own_only)} picks from your own sessions;"
                " pass --agents claude, codex or both with it."
            )
            sys.exit(2)
        if args.check:  # what --agents could import instead
            found = find_agents()
            report(found)
            names = ",".join(agent.name for agent in found if agent.sessions)
        if not has_key():
            print(f"[setup] MISSING: {NO_KEY}")
            sys.exit(2)
        print(
            f"[setup] Demo: sample Claude Code and Codex sessions; cognee runs on {configure_cognee()}."
        )
        if args.check:
            hint = f" Pass --agents (one or more of {names}) to import yours." if names else ""
            print(f"[setup] OK: ready to run the demo.{hint}")
            sys.exit(0)
        args.limit = DEFAULT_LIMIT
        run_demo(args)
        print(
            "[setup] Those answers come from sample sessions. To import your own, pass"
            " --agents claude, codex or both (--check lists what you have)."
        )
        sys.exit(0)

    if args.limit is None:
        args.limit = DEFAULT_LIMIT
    found = find_agents()
    report(found)
    missing = missing_setup(args, found)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    print(f"[setup] cognee runs on {configure_cognee()}.")
    print(f"[setup] Importing {' and '.join(AGENTS[agent] for agent in args.agents)} history.")
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)
    asyncio.run(run(args))
