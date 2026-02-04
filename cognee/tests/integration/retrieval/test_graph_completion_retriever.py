import os
import pytest
import pathlib
import pytest_asyncio
from typing import Optional, Union
import cognee

from cognee.low_level import setup, DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


@pytest_asyncio.fixture
async def setup_test_environment_simple():
    """Set up a clean test environment with simple graph data."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_graph_completion_context_simple")
    data_directory_path = str(base_dir / ".data_storage/test_graph_completion_context_simple")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    class Company(DataPoint):
        name: str
        description: str

    class Person(DataPoint):
        name: str
        description: str
        works_for: Company

    company1 = Company(name="Figma", description="Figma is a company")
    company2 = Company(name="Canva", description="Canvas is a company")
    person1 = Person(
        name="Steve Rodger",
        description="This is description about Steve Rodger",
        works_for=company1,
    )
    person2 = Person(
        name="Ike Loma", description="This is description about Ike Loma", works_for=company1
    )
    person3 = Person(
        name="Jason Statham",
        description="This is description about Jason Statham",
        works_for=company1,
    )
    person4 = Person(
        name="Mike Broski",
        description="This is description about Mike Broski",
        works_for=company2,
    )
    person5 = Person(
        name="Christina Mayer",
        description="This is description about Christina Mayer",
        works_for=company2,
    )

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
    system_directory_path = str(base_dir / ".cognee_system/test_graph_completion_context_complex")
    data_directory_path = str(base_dir / ".data_storage/test_graph_completion_context_complex")

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
        base_dir / ".cognee_system/test_get_graph_completion_context_on_empty_graph"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_get_graph_completion_context_on_empty_graph"
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
async def test_graph_completion_context_simple(setup_test_environment_simple):
    """Integration test: verify GraphCompletionRetriever can retrieve context (simple)."""
    retriever = GraphCompletionRetriever()
    query = "Who works at Canva?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    # Ensure the top-level sections are present
    assert "Nodes:" in context, "Missing 'Nodes:' section in context"
    assert "Connections:" in context, "Missing 'Connections:' section in context"

    # --- Nodes headers ---
    assert "Node: Steve Rodger" in context, "Missing node header for Steve Rodger"
    assert "Node: Figma" in context, "Missing node header for Figma"
    assert "Node: Ike Loma" in context, "Missing node header for Ike Loma"
    assert "Node: Jason Statham" in context, "Missing node header for Jason Statham"
    assert "Node: Mike Broski" in context, "Missing node header for Mike Broski"
    assert "Node: Canva" in context, "Missing node header for Canva"
    assert "Node: Christina Mayer" in context, "Missing node header for Christina Mayer"

    # --- Node contents ---
    assert (
        "__node_content_start__\nThis is description about Steve Rodger\n__node_content_end__"
        in context
    ), "Description block for Steve Rodger altered"
    assert "__node_content_start__\nFigma is a company\n__node_content_end__" in context, (
        "Description block for Figma altered"
    )
    assert (
        "__node_content_start__\nThis is description about Ike Loma\n__node_content_end__"
        in context
    ), "Description block for Ike Loma altered"
    assert (
        "__node_content_start__\nThis is description about Jason Statham\n__node_content_end__"
        in context
    ), "Description block for Jason Statham altered"
    assert (
        "__node_content_start__\nThis is description about Mike Broski\n__node_content_end__"
        in context
    ), "Description block for Mike Broski altered"
    assert "__node_content_start__\nCanvas is a company\n__node_content_end__" in context, (
        "Description block for Canva altered"
    )
    assert (
        "__node_content_start__\nThis is description about Christina Mayer\n__node_content_end__"
        in context
    ), "Description block for Christina Mayer altered"

    # --- Connections ---
    assert "Steve Rodger --[works_for]--> Figma" in context, (
        "Connection Steve Rodger→Figma missing or changed"
    )
    assert "Ike Loma --[works_for]--> Figma" in context, (
        "Connection Ike Loma→Figma missing or changed"
    )
    assert "Jason Statham --[works_for]--> Figma" in context, (
        "Connection Jason Statham→Figma missing or changed"
    )
    assert "Mike Broski --[works_for]--> Canva" in context, (
        "Connection Mike Broski→Canva missing or changed"
    )
    assert "Christina Mayer --[works_for]--> Canva" in context, (
        "Connection Christina Mayer→Canva missing or changed"
    )


@pytest.mark.asyncio
async def test_graph_completion_context_complex(setup_test_environment_complex):
    """Integration test: verify GraphCompletionRetriever can retrieve context (complex)."""
    retriever = GraphCompletionRetriever(top_k=20)
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert "Mike Rodger --[works_for]--> Figma" in context, "Failed to get Mike Rodger"
    assert "Ike Loma --[works_for]--> Figma" in context, "Failed to get Ike Loma"
    assert "Jason Statham --[works_for]--> Figma" in context, "Failed to get Jason Statham"


@pytest.mark.asyncio
async def test_get_graph_completion_context_on_empty_graph(setup_test_environment_empty):
    """Integration test: verify GraphCompletionRetriever handles empty graph correctly."""
    retriever = GraphCompletionRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)
    assert context == "", "Context should be empty on an empty graph"


@pytest.mark.asyncio
async def test_graph_completion_get_triplets_empty(setup_test_environment_empty):
    """Integration test: verify GraphCompletionRetriever get_triplets handles empty graph."""
    retriever = GraphCompletionRetriever()

    triplets = await retriever.get_triplets("Who works at Figma?")

    assert isinstance(triplets, list), "Triplets should be a list"
    assert len(triplets) == 0, "Should return empty list on empty graph"
