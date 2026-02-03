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
    """Set up a clean test environment with simple graph data."""
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
    """Set up a clean test environment with complex graph data."""
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
    """Set up a clean test environment without graph data."""
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
    query = "Who works at Canva?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert "Mike Broski --[works_for]--> Canva" in context, "Failed to get Mike Broski"
    assert "Christina Mayer --[works_for]--> Canva" in context, "Failed to get Christina Mayer"

    answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )


@pytest.mark.asyncio
async def test_graph_completion_extension_context_complex(setup_test_environment_complex):
    """Integration test: verify GraphCompletionContextExtensionRetriever can retrieve context (complex)."""
    retriever = GraphCompletionContextExtensionRetriever(top_k=20)
    query = "Who works at Figma and drives Tesla?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert "Mike Rodger --[works_for]--> Figma" in context, "Failed to get Mike Rodger"
    assert "Ike Loma --[works_for]--> Figma" in context, "Failed to get Ike Loma"
    assert "Jason Statham --[works_for]--> Figma" in context, "Failed to get Jason Statham"

    answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )


@pytest.mark.asyncio
async def test_get_graph_completion_extension_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify GraphCompletionContextExtensionRetriever handles empty graph correctly."""
    retriever = GraphCompletionContextExtensionRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)
    assert context == "", "Context should be empty on an empty graph"

    answer = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )

    assert isinstance(answer, list), f"Expected list, got {type(answer).__name__}"
    assert all(isinstance(item, str) and item.strip() for item in answer), (
        "Answer must contain only non-empty strings"
    )
