import os
import pytest
import pathlib
import pytest_asyncio
from typing import Optional, Union
import cognee

from cognee.low_level import setup, DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)


@pytest_asyncio.fixture
async def setup_test_environment_simple():
    """
    Prepare and yield a test environment populated with a simple person–company graph.
    
    Configures isolated system and data directories for the fixture, prunes existing data and system metadata, registers two DataPoint types (Company, Person), and persists sample entities: companies "Figma" and "Canva" and several Person instances linked via `works_for`. Yields control to the test, and on teardown attempts to prune data and system metadata.
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_graph_completion_extension_context_simple"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_graph_completion_extension_context_simple"
    )

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

    company1 = Company(name="Figma")
    company2 = Company(name="Canva")
    person1 = Person(name="Steve Rodger", works_for=company1)
    person2 = Person(name="Ike Loma", works_for=company1)
    person3 = Person(name="Jason Statham", works_for=company1)
    person4 = Person(name="Mike Broski", works_for=company2)
    person5 = Person(name="Christina Mayer", works_for=company2)

    entities = [company1, company2, person1, person2, person3, person4, person5]

    await add_data_points(entities)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_complex():
    """
    Set up a clean test environment populated with a complex graph of companies, people, vehicles, homes, and locations for integration tests.
    
    Creates two companies ("Figma" and "Canva") and multiple Person instances associated with those companies; some persons own Car and/or Home entities with Location data. Yields control to the test, and after the test completes it attempts to prune persisted data and system metadata to restore a clean state.
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_graph_completion_extension_context_complex"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_graph_completion_extension_context_complex"
    )

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    class Company(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class Car(DataPoint):
        brand: str
        model: str
        year: int

    class Location(DataPoint):
        country: str
        city: str

    class Home(DataPoint):
        location: Location
        rooms: int
        sqm: int

    class Person(DataPoint):
        name: str
        works_for: Company
        owns: Optional[list[Union[Car, Home]]] = None

    company1 = Company(name="Figma")
    company2 = Company(name="Canva")

    person1 = Person(name="Mike Rodger", works_for=company1)
    person1.owns = [Car(brand="Toyota", model="Camry", year=2020)]

    person2 = Person(name="Ike Loma", works_for=company1)
    person2.owns = [
        Car(brand="Tesla", model="Model S", year=2021),
        Home(location=Location(country="USA", city="New York"), sqm=80, rooms=4),
    ]

    person3 = Person(name="Jason Statham", works_for=company1)

    person4 = Person(name="Mike Broski", works_for=company2)
    person4.owns = [Car(brand="Ford", model="Mustang", year=1978)]

    person5 = Person(name="Christina Mayer", works_for=company2)
    person5.owns = [Car(brand="Honda", model="Civic", year=2023)]

    entities = [company1, company2, person1, person2, person3, person4, person5]

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
    Create and yield a clean, empty test environment for graph-related tests.
    
    Configures isolated system and data root directories for the fixture, removes all existing graph data and system metadata, runs initial setup, and yields control to the test. After the test completes, attempts to prune data and system metadata again and ignores any exceptions raised during teardown.
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_get_graph_completion_extension_context_on_empty_graph"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_get_graph_completion_extension_context_on_empty_graph"
    )

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_graph_completion_extension_context_simple(setup_test_environment_simple):
    """Integration test: verify GraphCompletionContextExtensionRetriever can retrieve context (simple)."""
    retriever = GraphCompletionContextExtensionRetriever()

    context = await resolve_edges_to_text(await retriever.get_context("Who works at Canva?"))

    assert "Mike Broski --[works_for]--> Canva" in context, "Failed to get Mike Broski"
    assert "Christina Mayer --[works_for]--> Canva" in context, "Failed to get Christina Mayer"

    answer = await retriever.get_completion("Who works at Canva?")

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )


@pytest.mark.asyncio
async def test_graph_completion_extension_context_complex(setup_test_environment_complex):
    """Integration test: verify GraphCompletionContextExtensionRetriever can retrieve context (complex)."""
    retriever = GraphCompletionContextExtensionRetriever(top_k=20)

    context = await resolve_edges_to_text(
        await retriever.get_context("Who works at Figma and drives Tesla?")
    )

    assert "Mike Rodger --[works_for]--> Figma" in context, "Failed to get Mike Rodger"
    assert "Ike Loma --[works_for]--> Figma" in context, "Failed to get Ike Loma"
    assert "Jason Statham --[works_for]--> Figma" in context, "Failed to get Jason Statham"

    answer = await retriever.get_completion("Who works at Figma?")

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )


@pytest.mark.asyncio
async def test_get_graph_completion_extension_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify GraphCompletionContextExtensionRetriever handles empty graph correctly."""
    retriever = GraphCompletionContextExtensionRetriever()

    context = await retriever.get_context("Who works at Figma?")
    assert context == [], "Context should be empty on an empty graph"

    answer = await retriever.get_completion("Who works at Figma?")

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )


@pytest.mark.asyncio
async def test_graph_completion_extension_get_triplets_empty(setup_test_environment_empty):
    """Integration test: verify GraphCompletionContextExtensionRetriever get_triplets handles empty graph."""
    retriever = GraphCompletionContextExtensionRetriever()

    triplets = await retriever.get_triplets("Who works at Figma?")

    assert isinstance(triplets, list), "Triplets should be a list"
    assert len(triplets) == 0, "Should return empty list on empty graph"