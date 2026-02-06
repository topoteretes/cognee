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
    """Set up a clean test environment with summaries."""
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
    """Set up a clean test environment without summaries."""
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
async def test_summaries_retriever(setup_test_environment_with_summaries):
    """Integration test: verify SummariesRetriever can retrieve summary context."""
    retriever = SummariesRetriever(top_k=20)
    query = "Christina"
    summaries = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=summaries)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=summaries, context=context
    )

    assert isinstance(completion, list), "Context should be a list"
    assert len(completion) > 0, "Context should not be empty"
    assert completion[0]["text"] == "C.M.", "Failed to get Christina Mayer"


@pytest.mark.asyncio
async def test_summaries_retriever_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify SummariesRetriever handles empty graph correctly."""
    retriever = SummariesRetriever()
    query = "Christina Mayer"

    with pytest.raises(NoDataError):
        await retriever.get_retrieved_objects(query)

    vector_engine = get_vector_engine()
    await vector_engine.create_collection("TextSummary_text", payload_schema=TextSummary)

    summaries = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=summaries)
    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=summaries, context=context
    )

    assert completion == [], "Returned context should be empty on an empty graph"
