"""Select repeated nested JSON arrays when building a knowledge graph."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cognee

from cognee.modules.chunking.JsonListChunker import JsonListChunker


async def main():
    data = {
        "tenant": "ACME",
        "records": [
            {"source": "sensor-A", "items": [{"temperature": 21.4}]},
            {"source": "sensor-B", "items": [{"temperature": 22.1}]},
        ],
    }

    with TemporaryDirectory() as directory:
        data_path = Path(directory) / "readings.json"
        data_path.write_text(json.dumps(data), encoding="utf-8")

        await cognee.add(str(data_path), dataset_name="nested_json_example")
        await cognee.cognify(
            datasets="nested_json_example",
            chunker=JsonListChunker.with_json_path("records[*].items"),
        )


if __name__ == "__main__":
    asyncio.run(main())
