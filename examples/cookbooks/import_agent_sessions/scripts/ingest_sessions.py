"""Remember the session files that read_sessions.py wrote, one node set per agent.

All sessions go into one cognee dataset (`imported_agent_sessions`), so a question sees every agent's
history at once. Each file is tagged with its agent's node set (`<dataset>:claude_code` or
`<dataset>:codex`), so
ask.py can also answer from one agent's history alone. Running it again adds new sessions,
skips files whose content did not change, and replaces a session that went on since the last
import: cognee keys a file by its content, so the earlier copy is forgotten first.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/ingest_sessions.py \
    [--agents claude,codex]
"""

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.engine.operations.setup import setup

try:
    from scripts.configure import configure_cognee
except ModuleNotFoundError:  # run alone from scripts/
    from configure import configure_cognee

DATASET = "imported_agent_sessions"  # the same in every script
SESSIONS = Path(__file__).parent.parent / "sessions"
AGENT_NODE_SETS = {"claude": "claude_code", "codex": "codex"}


def node_set(dataset: str, agent: str) -> str:
    """The agent's node set, named after the dataset too. With access control off, every
    dataset shares one graph, and only node sets keep this dataset's sessions apart."""
    return f"{dataset}:{AGENT_NODE_SETS[agent]}"


async def forget_earlier_copies(dataset: str, files: list[str]) -> int:
    """Forget the imported copy of each session whose file has changed since (the session went
    on). Files are named after the session, so the copy has the file's name and another hash."""
    await setup()  # a fresh install has no database yet
    found = [d for d in await cognee.datasets.list_datasets() if d.name == dataset]
    if not found:
        return 0
    hashes = {Path(f).stem: hashlib.md5(Path(f).read_bytes()).hexdigest() for f in files}
    stale = [
        data
        for data in await cognee.datasets.list_data(found[0].id)
        if data.name in hashes and data.content_hash != hashes[data.name]
    ]
    for data in stale:
        await cognee.forget(data_id=data.id, dataset_id=found[0].id)
    return len(stale)


async def ingest_sessions(
    agents: list[str], dataset: str = DATASET, sessions: Path = SESSIONS
) -> None:
    for agent in agents:
        files = sorted(str(path) for path in (sessions / agent).glob("*.md"))
        if not files:
            print(f"[ingest_sessions] No {agent} sessions in {sessions / agent}; skipped.")
            continue
        replaced = await forget_earlier_copies(dataset, files)
        if replaced:
            print(
                f"[ingest_sessions] {replaced} {agent} sessions went on since the last import;"
                " their earlier copies were forgotten"
            )
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
    args = parser.parse_args()
    configure_cognee()
    asyncio.run(ingest_sessions(args.agents.split(",")))
