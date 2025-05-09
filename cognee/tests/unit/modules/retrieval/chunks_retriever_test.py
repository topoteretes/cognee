import os
import pytest
import pathlib

import cognee
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever


class TestChunksRetriever:
    @pytest.mark.asyncio
    async def test_chunk_context_simple(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunks_context_simple"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunks_context_simple"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Steve Rodger's career",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Steve Rodger",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
        )
        chunk2 = DocumentChunk(
            text="Mike Broski",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
        )
        chunk3 = DocumentChunk(
            text="Christina Mayer",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
        )

        entities = [chunk1, chunk2, chunk3]

        await add_data_points(entities)

        retriever = ChunksRetriever()

        context = await retriever.get_context("Mike")

        assert context[0]["text"] == "Mike Broski", "Failed to get Mike Broski"

    @pytest.mark.asyncio
    async def test_chunk_context_complex(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunk_context_complex"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunk_context_complex"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document1 = TextDocument(
            name="Employee List",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
        )

        document2 = TextDocument(
            name="Car List",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Steve Rodger",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk2 = DocumentChunk(
            text="Mike Broski",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk3 = DocumentChunk(
            text="Christina Mayer",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )

        chunk4 = DocumentChunk(
            text="Range Rover",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk5 = DocumentChunk(
            text="Hyundai",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk6 = DocumentChunk(
            text="Chrysler",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )

        entities = [chunk1, chunk2, chunk3, chunk4, chunk5, chunk6]

        await add_data_points(entities)

        retriever = ChunksRetriever(top_k=20)

        context = await retriever.get_context("Christina")

        assert context[0]["text"] == "Christina Mayer", "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_chunk_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunk_context_empty"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunk_context_empty"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = ChunksRetriever()

        with pytest.raises(NoDataError):
            await retriever.get_context("Christina Mayer")

        vector_engine = get_vector_engine()
        await vector_engine.create_collection("DocumentChunk_text", payload_schema=DocumentChunk)

        context = await retriever.get_context("Christina Mayer")
        assert len(context) == 0, "Found chunks when none should exist"


if __name__ == "__main__":
    from asyncio import run

    test = TestChunksRetriever()

    async def main():
        await test.test_chunk_context_simple()
        await test.test_chunk_context_complex()
        await test.test_chunk_context_on_empty_graph()

    run(main())
