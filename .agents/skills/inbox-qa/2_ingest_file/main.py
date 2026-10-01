"""Step 2: remember a text file (notes.txt in this folder by default).

Run alone: uv run python .agents/skills/inbox-qa/2_ingest_file/main.py [file.txt]
"""

import asyncio
import sys
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "inbox_qa_skill"  # the same in every step
SAMPLE_FILE = Path(__file__).parent / "notes.txt"


async def main(path: Path) -> None:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"[ingest_file] MISSING: no such file {path}")
    await cognee.remember(
        str(path), dataset_name=DATASET, node_set=["notes"], self_improvement=False
    )
    print(f"[ingest_file] Remembered: {path.name}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_FILE))
