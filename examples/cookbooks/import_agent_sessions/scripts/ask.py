"""Ask a question about your imported agent sessions.

The question is answered once over every imported session, then once per agent from that
agent's node set alone, so you see what each agent's history says and what they say together.
Each answer runs in a new cognee session: without one, recall shares the default session, and
the answers from earlier runs would end up in this one's context. Automatic feedback analysis is
off here, since nothing ever improves those sessions.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/ask.py \
    "What did I work on in these sessions?" [--agents claude,codex]
"""

import argparse
import asyncio
import os
import uuid

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.search.types import SearchType

# Nothing improves these one-answer sessions, so skip the per-answer feedback analysis (one more
# LLM call each). Set after importing cognee, which loads .env, so it wins over a .env value.
os.environ["AUTO_FEEDBACK"] = "false"

try:
    from scripts.configure import configure_cognee
    from scripts.ingest_sessions import node_set
except ModuleNotFoundError:  # run alone from scripts/
    from configure import configure_cognee
    from ingest_sessions import node_set

DATASET = "imported_agent_sessions"  # the same in every script
LABELS = {"claude": "Claude Code", "codex": "Codex"}

PROMPT = """You answer questions about the user's own coding-agent sessions (Claude Code and
Codex), which are in the context. Name projects, tickets and files when they matter. Answer in
two to five sentences, and say so when the sessions don't cover something."""


async def answer(question: str, dataset: str, node_name: list[str] | None = None) -> str:
    results = await cognee.recall(
        question,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[dataset],
        node_name=node_name,
        session_id=f"ask-{uuid.uuid4()}",  # a new session, so earlier answers don't leak in
        system_prompt=PROMPT,
    )
    return str(results[0].text) if results else "Nothing found."


async def ask(question: str, agents: list[str], dataset: str = DATASET) -> None:
    every_agent = [node_set(dataset, agent) for agent in agents]  # any of them (OR)
    print(f"[ask] you> {question}")
    print(f"[ask] all sessions> {await answer(question, dataset, every_agent)}")
    if len(agents) > 1:
        for agent in agents:
            print(
                f"[ask] {LABELS[agent]} only> {await answer(question, dataset, [node_set(dataset, agent)])}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--agents", default="claude,codex", help="claude, codex or both")
    args = parser.parse_args()
    configure_cognee()
    asyncio.run(ask(args.question, args.agents.split(",")))
