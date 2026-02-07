import os
from typing import List
import pytest
import pathlib
import pytest_asyncio
import cognee

from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity


class DocumentChunkWithEntities(DataPoint):
    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    contains: List[Entity] = None

    metadata: dict = {"index_fields": ["text"]}


@pytest_asyncio.fixture
async def setup_test_environment_with_chunks_simple():
    """Set up a clean test environment with simple chunks."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_rag_completion_context_simple")
    data_directory_path = str(base_dir / ".data_storage/test_rag_completion_context_simple")

    cognee.config.system_root_directory(system_directory_path)
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

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )

        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_with_chunks_complex():
    """Set up a clean test environment with complex chunks."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_rag_completion_context_complex")
    data_directory_path = str(base_dir / ".data_storage/test_rag_completion_context_complex")

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

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )

        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_empty():
    """Set up a clean test environment without chunks."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_get_rag_completion_context_on_empty_graph"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_get_rag_completion_context_on_empty_graph"
    )

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )

        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_rag_completion_context_simple(setup_test_environment_with_chunks_simple):
    """Integration test: verify CompletionRetriever can retrieve context (simple)."""
    retriever = CompletionRetriever()
    query = "Mike"

    retrieved_objects = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(
        query=query, retrieved_objects=retrieved_objects
    )

    assert isinstance(context, str), "Context should be a string"
    assert "Mike Broski" in context, "Failed to get Mike Broski"


@pytest.mark.asyncio
async def test_rag_completion_context_multiple_chunks(setup_test_environment_with_chunks_simple):
    """Integration test: verify CompletionRetriever can retrieve context from multiple chunks."""
    retriever = CompletionRetriever()
    query = "Steve"

    retrieved_objects = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(
        query=query, retrieved_objects=retrieved_objects
    )

    assert isinstance(context, str), "Context should be a string"
    assert "Steve Rodger" in context, "Failed to get Steve Rodger"


@pytest.mark.asyncio
async def test_rag_completion_context_complex(setup_test_environment_with_chunks_complex):
    """Integration test: verify CompletionRetriever can retrieve context (complex)."""
    # TODO: top_k doesn't affect the output, it should be fixed.
    retriever = CompletionRetriever(top_k=20)
    query = "Christina"

    retrieved_objects = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(
        query=query, retrieved_objects=retrieved_objects
    )

    assert context[0:15] == "Christina Mayer", "Failed to get Christina Mayer"


@pytest.mark.asyncio
async def test_get_rag_completion_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify CompletionRetriever handles empty graph correctly."""
    retriever = CompletionRetriever()
    query = "Christina Mayer"

    with pytest.raises(NoDataError):
        await retriever.get_retrieved_objects(query)

    vector_engine = get_vector_engine()
    await vector_engine.create_collection(
        "DocumentChunk_text", payload_schema=DocumentChunkWithEntities
    )

    retrieved_objects = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(
        query=query, retrieved_objects=retrieved_objects
    )
    assert context == "", "Returned context should be empty on an empty graph"
