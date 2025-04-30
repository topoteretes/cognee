import os
import pytest
import pathlib

import cognee
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.tasks.summarization.models import TextSummary
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever


class TextSummariesRetriever:
    @pytest.mark.asyncio
    async def test_chunk_context(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_summary_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_summary_context"
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
        chunk1_summary = TextSummary(
            text="S.R.",
            made_from=chunk1,
        )
        chunk2 = DocumentChunk(
            text="Mike Broski",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk2_summary = TextSummary(
            text="M.B.",
            made_from=chunk2,
        )
        chunk3 = DocumentChunk(
            text="Christina Mayer",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk3_summary = TextSummary(
            text="C.M.",
            made_from=chunk3,
        )
        chunk4 = DocumentChunk(
            text="Range Rover",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk4_summary = TextSummary(
            text="R.R.",
            made_from=chunk4,
        )
        chunk5 = DocumentChunk(
            text="Hyundai",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk5_summary = TextSummary(
            text="H.Y.",
            made_from=chunk5,
        )
        chunk6 = DocumentChunk(
            text="Chrysler",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk6_summary = TextSummary(
            text="C.H.",
            made_from=chunk6,
        )

        entities = [
            chunk1_summary,
            chunk2_summary,
            chunk3_summary,
            chunk4_summary,
            chunk5_summary,
            chunk6_summary,
        ]

        await add_data_points(entities)

        retriever = SummariesRetriever(top_k=20)

        context = await retriever.get_context("Christina")

        assert context[0]["text"] == "C.M.", "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_chunk_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_summary_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_summary_context"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = SummariesRetriever()

        with pytest.raises(NoDataError):
            await retriever.get_context("Christina Mayer")

        vector_engine = get_vector_engine()
        await vector_engine.create_collection("TextSummary_text", payload_schema=TextSummary)

        context = await retriever.get_context("Christina Mayer")
        assert context == [], "Returned context should be empty on an empty graph"


if __name__ == "__main__":
    from asyncio import run

    test = TextSummariesRetriever()

    run(test.test_chunk_context())
    run(test.test_chunk_context_on_empty_graph())
