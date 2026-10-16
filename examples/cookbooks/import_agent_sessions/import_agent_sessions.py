"""Import your Claude Code and Codex session history into cognee, then ask about it.

    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --demo
    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py --check
    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py
    uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py \
        --limit 50 --project cognee --ask "Which tickets did I work on?"

It first lists the coding agents installed on this machine and the sessions each has kept,
then asks which agents' sessions to ingest (or takes them from --agents; without a terminal
to ask in, --agents is required). One LLM key runs cognee and imports every agent's history:
an OpenAI or an Anthropic key, or the LLM_API_KEY your .env configures. Everything goes into
one dataset, so every question is answered over all the agents you chose. --demo instead
imports the sample Claude Code and Codex sessions from setup.py into a dataset of its own, so
it works with no history. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup
missing or no agents chosen, 1 a script failed (its traceback says why).
"""

import argparse
import asyncio
import os
import sys

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.ask import ask
from scripts.clear import clear
from scripts.find_agents import Agent, describe, find_agents, not_read
from scripts.ingest_sessions import DATASET, ingest_sessions
from scripts.read_sessions import AGENTS, OUT, read_sessions, write_sessions
from setup import SAMPLE, write_sample

DEFAULT_QUESTION = "What is the topic of my data?"
DEMO_DATASET = "imported_agent_sessions_demo"  # never mixes with your own imported sessions
ANTHROPIC_MODEL = "claude-haiku-5-5"  # cheap and fast; cognify makes one call per chunk


NO_KEY = "No LLM key is set. Put OPENAI_API_KEY or ANTHROPIC_API_KEY in .env."


def has_key() -> bool:
    """Whether cognee has an LLM key to run on. Any one key imports every agent's history."""
    return any(
        os.environ.get(name) for name in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )


def configure_cognee() -> str:
    """Run cognee on a key you have, and say which. An LLM_API_KEY in .env is used as it is."""
    import cognee

    if os.environ.get("LLM_API_KEY"):
        return f"{os.environ.get('LLM_PROVIDER', 'openai')} (LLM_API_KEY)"
    if os.environ.get("OPENAI_API_KEY"):
        # embeddings reuse it, as by default
        cognee.config.set_llm_api_key(os.environ["OPENAI_API_KEY"])
        return "openai (OPENAI_API_KEY)"
    cognee.config.set_llm_config(
        {
            "llm_provider": "anthropic",
            "llm_model": ANTHROPIC_MODEL,
            "llm_api_key": os.environ["ANTHROPIC_API_KEY"],
        }
    )
    if not os.environ.get("EMBEDDING_PROVIDER"):
        # Anthropic has no embedding API, so embed on your machine with fastembed (a core
        # dependency; the model downloads once, about 70 MB)
        cognee.config.set_embedding_config(
            {
                "embedding_provider": "fastembed",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimensions": 384,
            }
        )
    return f"anthropic {ANTHROPIC_MODEL} (ANTHROPIC_API_KEY)"


def choose_agents(choices: list[Agent]) -> list[str]:
    """Ask in the terminal which agents' sessions to ingest. Enter takes all of them."""
    print("[choose] Which agent sessions should be ingested into cognee?")
    for number, agent in enumerate(choices, 1):
        print(f"[choose]   {number}) {AGENTS[agent.name]}: {agent.sessions} sessions")
    while True:
        reply = input("[choose] Numbers separated by commas, or Enter for all: ").strip()
        if not reply or reply.lower() == "all":
            return [agent.name for agent in choices]
        try:
            numbers = [int(part) for part in reply.split(",")]
            if not all(1 <= number <= len(choices) for number in numbers):
                raise ValueError(reply)
        except ValueError:
            print(f"[choose] Type numbers from 1 to {len(choices)}, like 1 or 1,2.")
            continue
        return list(dict.fromkeys(choices[n - 1].name for n in numbers))  # in order, no repeats


def missing_setup(args: argparse.Namespace, found: list[Agent]) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    with_sessions = [agent.name for agent in found if agent.sessions]
    if not with_sessions:
        return ["No Claude Code or Codex sessions on this machine, so there is nothing to import."]
    if not has_key():
        return [NO_KEY]
    missing = []
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
    parser.add_argument("--demo", action="store_true", help="import the sample sessions instead")
    parser.add_argument(
        "--agents",
        help="claude, codex or both (default: ask which, from the agents with sessions and a key)",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="newest sessions per agent (default 20)"
    )
    parser.add_argument("--project", help="only sessions whose folder path contains this text")
    parser.add_argument("--since", help="only sessions started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--clear", action="store_true", help="forget the dataset before importing")
    parser.add_argument("--ask", default=DEFAULT_QUESTION, help=f'default: "{DEFAULT_QUESTION}"')
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if args.demo:  # the sample has both agents' sessions
        if not has_key():
            print(f"[setup] MISSING: {NO_KEY}")
            sys.exit(2)
        print(
            f"[setup] Demo: sample Claude Code and Codex sessions; cognee runs on {configure_cognee()}."
        )
        if args.check:
            print("[setup] OK: ready to run.")
            sys.exit(0)
        run_demo(args)
        sys.exit(0)

    found = find_agents()
    for agent in found:
        print(f"[find_agents] {describe(agent)}")
    for line in not_read():
        print(f"[find_agents] Not imported: {line}.")

    choices = [agent for agent in found if agent.sessions]
    args.agents = args.agents.split(",") if args.agents else []
    missing = missing_setup(args, found)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    print(f"[setup] cognee runs on {configure_cognee()}.")

    if not args.agents:
        names = ",".join(agent.name for agent in choices)
        if args.check or not sys.stdin.isatty():
            for agent in choices:
                print(f"[setup] Can import: {agent.name} ({agent.sessions} sessions)")
        if args.check:
            print(f"[setup] OK: ready to run. Choose with --agents (one or more of {names}).")
            sys.exit(0)
        if not sys.stdin.isatty():  # no one to ask, as when an agent runs this
            print(f"[setup] CHOOSE: pass --agents with one or more of {names}.")
            sys.exit(2)
        args.agents = choose_agents(choices)
    print(f"[setup] Importing {' and '.join(AGENTS[agent] for agent in args.agents)} history.")
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)
    asyncio.run(run(args))
