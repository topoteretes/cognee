"""Remember the session files that read_sessions.py wrote, one node set per agent.

All sessions go into one cognee dataset (`imported_agent_sessions`), so a question sees every agent's
history at once. Each file is tagged with its agent's node set (`<dataset>:claude_code` or
`<dataset>:codex`), so
ask.py can also answer from one agent's history alone. Running it again adds new sessions
and skips files whose content did not change.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/ingest_sessions.py \
    [--agents claude,codex]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee

DATASET = "imported_agent_sessions"  # the same in every script
SESSIONS = Path(__file__).parent.parent / "sessions"
AGENT_NODE_SETS = {"claude": "claude_code", "codex": "codex"}


def node_set(dataset: str, agent: str) -> str:
    """The agent's node set, named after the dataset too. With access control off, every
    dataset shares one graph, and only node sets keep this dataset's sessions apart."""
    return f"{dataset}:{AGENT_NODE_SETS[agent]}"


async def ingest_sessions(
    agents: list[str], dataset: str = DATASET, sessions: Path = SESSIONS
) -> None:
    for agent in agents:
        files = sorted(str(path) for path in (sessions / agent).glob("*.md"))
        if not files:
            print(f"[ingest_sessions] No {agent} sessions in {sessions / agent}; skipped.")
            continue
        await cognee.remember(
            files,
            dataset_name=dataset,
            node_set=[node_set(dataset, agent)],
            self_improvement=False,
        )
        print(
            f"[ingest_sessions] Remembered {len(files)} {agent} sessions as {node_set(dataset, agent)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", default="claude,codex", help="claude, codex or both")
    asyncio.run(ingest_sessions(parser.parse_args().agents.split(",")))
