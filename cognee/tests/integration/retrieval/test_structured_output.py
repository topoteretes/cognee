import asyncio
import os
import pytest
import pathlib
import pytest_asyncio
import cognee

from pydantic import BaseModel
from cognee.low_level import setup, DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.retrieval.entity_extractors.DummyEntityExtractor import DummyEntityExtractor
from cognee.modules.retrieval.context_providers.DummyContextProvider import DummyContextProvider
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.EntityCompletionRetriever import EntityCompletionRetriever
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever


class TestAnswer(BaseModel):
    answer: str
    explanation: str


def _assert_string_answer(answer: list[str]):
    assert isinstance(answer, list), f"Expected str, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), "Items should be strings"
    assert all(item.strip() for item in answer), "Items should not be empty"


def _assert_structured_answer(answer: list[TestAnswer]):
    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(x, TestAnswer) for x in answer), "Items should be TestAnswer"
    assert all(x.answer.strip() for x in answer), "Answer text should not be empty"
    assert all(x.explanation.strip() for x in answer), "Explanation should not be empty"


async def _test_get_structured_graph_completion_cot():
    retriever = GraphCompletionCotRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    triplets = await retriever.get_retrieved_objects(query)
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_structured_answer(structured_answer)


async def _test_get_structured_graph_completion():
    retriever = GraphCompletionRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_structured_answer(structured_answer)


async def _test_get_structured_graph_completion_temporal():
    retriever = TemporalRetriever()
    query = "When did Steve start working at Figma?"

    # Test with string response model (default)
    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_structured_answer(structured_answer)


async def _test_get_structured_graph_completion_rag():
    retriever = CompletionRetriever()
    query = "Where does Steve work?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_structured_answer(structured_answer)


async def _test_get_structured_graph_completion_context_extension():
    retriever = GraphCompletionContextExtensionRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )
    _assert_structured_answer(structured_answer)


async def _test_get_structured_entity_completion():
    retriever = EntityCompletionRetriever(DummyEntityExtractor(), DummyContextProvider())
    query = "Who is Albert Einstein?"

    entities = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=entities)

    # Test with string response model (default)
    string_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=entities, context=context
    )
    _assert_string_answer(string_answer)

    retriever.response_model = TestAnswer
    # Test with structured response model
    structured_answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=entities, context=context
    )
    _assert_structured_answer(structured_answer)


@pytest_asyncio.fixture
async def setup_test_environment():
    """Set up a clean test environment with graph and document data."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_get_structured_completion")
    data_directory_path = str(base_dir / ".data_storage/test_get_structured_completion")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    class Company(DataPoint):
        name: str

    class Person(DataPoint):
        name: str
        works_for: Company
        works_since: int

    company1 = Company(name="Figma")
    person1 = Person(name="Steve Rodger", works_for=company1, works_since=2015)

    entities = [company1, person1]
    await add_data_points(entities)

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

    entity_type = EntityType(name="Person", description="A human individual")
    entity = Entity(name="Albert Einstein", is_a=entity_type, description="A famous physicist")

    entities = [entity]
    await add_data_points(entities)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_get_structured_completion(setup_test_environment):
    """Integration test: verify structured output completion for all retrievers."""
    await _test_get_structured_graph_completion_cot()
    await _test_get_structured_graph_completion()
    await _test_get_structured_graph_completion_temporal()
    await _test_get_structured_graph_completion_rag()
    await _test_get_structured_graph_completion_context_extension()
    await _test_get_structured_entity_completion()
