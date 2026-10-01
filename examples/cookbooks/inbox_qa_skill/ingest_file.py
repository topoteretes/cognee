"""Step 2: remember a text file (the sample notes by default).

Run alone: uv run python examples/cookbooks/inbox_qa_skill/ingest_file.py [path/to/file.txt]
"""

import asyncio
import sys
from pathlib import Path

from common import DATASET, SAMPLE_FILE

import cognee


async def run(path: Path = SAMPLE_FILE) -> None:
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"No such file: {path}")
    await cognee.remember(
        str(path), dataset_name=DATASET, node_set=["notes"], self_improvement=False
    )
    print(f"[ingest_file] Remembered: {path.name}")


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_FILE))
