import asyncio
import cognee
from cognee.shared.logging_utils import setup_logging, INFO
from cognee.temporal_poc.temporal_cognify import temporal_cognify

import json
from pathlib import Path


async def reading_temporal_data():
    path = Path("cognee/temporal_poc/test_hard.json")
    contexts = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ctx = entry.get("context", "")
            if ctx and ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
    return contexts


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    texts = await reading_temporal_data()
    texts = texts[:10]

    await cognee.add(texts)
    await temporal_cognify()

    print()


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
