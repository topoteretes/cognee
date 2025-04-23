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
from cognee.modules.retrieval.completion_retriever import CompletionRetriever


class TestRAGCompletionRetriever:
    @pytest.mark.asyncio
    async def test_rag_completion_context_simple(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_rag_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_rag_context"
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

        retriever = CompletionRetriever()

        context = await retriever.get_context("Mike")

        assert context == "Mike Broski", "Failed to get Mike Broski"

    @pytest.mark.asyncio
    async def test_rag_completion_context_complex(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_graph_completion_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_graph_completion_context"
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

        # TODO: top_k doesn't affect the output, it should be fixed.
        retriever = CompletionRetriever(top_k=20)

        context = await retriever.get_context("Christina")

        assert context[0:15] == "Christina Mayer", "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_get_rag_completion_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_graph_completion_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_graph_completion_context"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = CompletionRetriever()

        with pytest.raises(NoDataError):
            await retriever.get_context("Christina Mayer")

        vector_engine = get_vector_engine()
        await vector_engine.create_collection("DocumentChunk_text", payload_schema=DocumentChunk)

        context = await retriever.get_context("Christina Mayer")
        assert context == "", "Returned context should be empty on an empty graph"


if __name__ == "__main__":
    from asyncio import run

    test = TestRAGCompletionRetriever()

    run(test.test_rag_completion_context_simple())
    run(test.test_rag_completion_context_complex())
    run(test.test_get_rag_completion_context_on_empty_graph())
