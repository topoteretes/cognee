import asyncio
from uuid import uuid4
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
import time
import uuid
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import PdfDocument


def create_node(name):
    document = PdfDocument(
        id=uuid.uuid4(),
        name=name,
        raw_data_location=name,
        external_metadata="",
        mime_type="",
    )
    return document

async def main():
    adapter = KuzuAdapter("test.db")
    nodes = [create_node(f"Node{i}") for i in range(200000)]

    print("Writer: Starting...")
    await adapter.add_nodes(nodes)

    print("writer finished...")

    time.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())