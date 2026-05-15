import asyncio
import time
import uuid

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.modules.data.processing.document_types import PdfDocument
from common import get_kuzu_db_path


def create_node(name):
    return PdfDocument(
        id=uuid.uuid4(),
        name=name,
        raw_data_location=name,
        external_metadata="test_external_metadata",
        mime_type="test_mime",
    )


async def main():
    adapter = LadybugAdapter(get_kuzu_db_path())
    nodes = [create_node(f"Node{i}") for i in range(5)]

    print("Writer: Starting...")
    await adapter.add_nodes(nodes)
    print("writer finished...")

    time.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
