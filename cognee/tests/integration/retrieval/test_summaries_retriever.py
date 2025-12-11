import os
import pytest
import pathlib
import pytest_asyncio
import cognee

from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.tasks.summarization.models import TextSummary
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever


@pytest_asyncio.fixture
async def setup_test_environment_with_summaries():
    """
    Prepare a fresh test environment populated with six TextSummary entities linked to two sample TextDocument records.
    
    Configures test-specific system and data root directories, removes any existing data and system metadata, performs low-level setup, creates two documents with three DocumentChunk entries each and corresponding TextSummary entities, and adds those summaries to the data store. Yields control to the test, then attempts to prune data and system metadata as teardown, ignoring any cleanup errors.
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_summaries_retriever_context")
    data_directory_path = str(base_dir / ".data_storage/test_summaries_retriever_context")

    cognee.config.system_root_directory(system_directory_path)
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

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_empty():
    """
    Prepare a clean test environment configured for an empty summaries graph.
    
    Initializes Cognee system and data root directories for the empty test context, prunes any existing data and system metadata before the test, yields control to the test, and attempts to prune data and system metadata again during teardown (ignoring any errors).
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_summaries_retriever_context_empty")
    data_directory_path = str(base_dir / ".data_storage/test_summaries_retriever_context_empty")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_summaries_retriever_context(setup_test_environment_with_summaries):
    """Integration test: verify SummariesRetriever can retrieve summary context."""
    retriever = SummariesRetriever(top_k=20)

    context = await retriever.get_context("Christina")

    assert isinstance(context, list), "Context should be a list"
    assert len(context) > 0, "Context should not be empty"
    assert context[0]["text"] == "C.M.", "Failed to get Christina Mayer"


@pytest.mark.asyncio
async def test_summaries_retriever_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify SummariesRetriever handles empty graph correctly."""
    retriever = SummariesRetriever()

    with pytest.raises(NoDataError):
        await retriever.get_context("Christina Mayer")

    vector_engine = get_vector_engine()
    await vector_engine.create_collection("TextSummary_text", payload_schema=TextSummary)

    context = await retriever.get_context("Christina Mayer")
    assert context == [], "Returned context should be empty on an empty graph"