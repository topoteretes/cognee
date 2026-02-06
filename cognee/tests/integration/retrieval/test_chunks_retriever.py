import os
import pytest
import pathlib
import pytest_asyncio
from typing import List
import cognee

from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
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
    system_directory_path = str(base_dir / ".cognee_system/test_chunks_retriever_context_simple")
    data_directory_path = str(base_dir / ".data_storage/test_chunks_retriever_context_simple")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

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
    system_directory_path = str(base_dir / ".cognee_system/test_chunks_retriever_context_complex")
    data_directory_path = str(base_dir / ".data_storage/test_chunks_retriever_context_complex")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

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
    system_directory_path = str(base_dir / ".cognee_system/test_chunks_retriever_context_empty")
    data_directory_path = str(base_dir / ".data_storage/test_chunks_retriever_context_empty")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

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
async def test_chunks_retriever_multiple_chunks(setup_test_environment_with_chunks_simple):
    """Integration test: verify ChunksRetriever can retrieve multiple chunks."""
    retriever = ChunksRetriever()
    query = "Steve"
    chunks = await retriever.get_retrieved_objects("Steve")
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=chunks)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=chunks, context=context
    )

    assert isinstance(completion, list), "Retrieved objects should be a list"
    assert len(completion) > 0, "Retrieved objects list should not be empty"
    assert any(chunk["text"] == "Steve Rodger" for chunk in completion), (
        "Failed to get Steve Rodger chunk"
    )


@pytest.mark.asyncio
async def test_chunks_retriever_top_k_limit(setup_test_environment_with_chunks_complex):
    """Integration test: verify ChunksRetriever respects top_k parameter."""
    retriever = ChunksRetriever(top_k=2)
    query = "Employee"

    chunks = await retriever.get_retrieved_objects("Steve")
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=chunks)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=chunks, context=context
    )

    assert isinstance(completion, list), "Context should be a list"
    assert len(completion) <= 2, "Should respect top_k limit"


@pytest.mark.asyncio
async def test_chunks_retriever_context_complex(setup_test_environment_with_chunks_complex):
    """Integration test: verify ChunksRetriever can retrieve chunk context (complex)."""
    retriever = ChunksRetriever(top_k=20)
    query = "Christina"

    chunks = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=chunks)

    assert context[0:15] == "Christina Mayer", "Failed to get Christina Mayer"


@pytest.mark.asyncio
async def test_chunks_retriever_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify ChunksRetriever handles empty graph correctly."""
    retriever = ChunksRetriever()
    query = "Christina Mayer"

    vector_engine = get_vector_engine()
    await vector_engine.create_collection(
        "DocumentChunk_text", payload_schema=DocumentChunkWithEntities
    )

    chunks = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=chunks)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=chunks, context=context
    )
    assert isinstance(completion, list), "Retrieved objects should be a list"
    assert len(completion) == 0, "Found chunks when none should exist"


@pytest.mark.asyncio
async def test_chunks_retriever_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify ChunksRetriever context handles empty graph correctly."""
    retriever = ChunksRetriever()
    query = "Christina Mayer"

    vector_engine = get_vector_engine()
    await vector_engine.create_collection(
        "DocumentChunk_text", payload_schema=DocumentChunkWithEntities
    )

    chunks = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=chunks)
    assert context == "", "Found chunks when none should exist"
