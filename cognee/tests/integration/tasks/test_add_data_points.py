import pathlib
import pytest
import pytest_asyncio

import cognee
from cognee.low_level import setup
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tasks.storage.exceptions import InvalidDataPointsInAddDataPointsError
from cognee.infrastructure.databases.graph import get_graph_engine


class Person(DataPoint):
    name: str
    age: int
    metadata: dict = {"index_fields": ["name"]}


class Company(DataPoint):
    name: str
    industry: str
    metadata: dict = {"index_fields": ["name", "industry"]}


class Document(DataPoint):
    title: str
    content: str
    metadata: dict = {"index_fields": ["title", "content"]}


@pytest_asyncio.fixture
async def clean_test_environment():
    """Set up a clean test environment for add_data_points tests."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_add_data_points_integration")
    data_directory_path = str(base_dir / ".data_storage/test_add_data_points_integration")

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
async def test_add_simple_data_points(clean_test_environment):
    """Integration test: add simple data points to graph database."""
    person1 = Person(name="Alice", age=30)
    person2 = Person(name="Bob", age=25)

    result = await add_data_points([person1, person2])

    assert result == [person1, person2]
    assert len(result) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 2


@pytest.mark.asyncio
async def test_add_empty_list(clean_test_environment):
    """Integration test: adding empty list should not error."""
    result = await add_data_points([])

    assert result == []

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 0


@pytest.mark.asyncio
async def test_add_data_points_with_custom_edges(clean_test_environment):
    """Integration test: add custom edges between data points."""
    person1 = Person(name="Charlie", age=35)
    person2 = Person(name="Diana", age=32)

    custom_edge = (str(person1.id), str(person2.id), "knows", {"edge_text": "friends with"})

    result = await add_data_points([person1, person2], custom_edges=[custom_edge])

    assert len(result) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(edges) >= 1


@pytest.mark.asyncio
async def test_add_data_points_with_relationships(clean_test_environment):
    """Integration test: relationships create edges automatically."""

    class Employee(DataPoint):
        name: str
        works_at: Company
        metadata: dict = {"index_fields": ["name"]}

    company = Company(name="TechCorp", industry="Technology")
    employee = Employee(name="Eve", works_at=company)

    result = await add_data_points([employee])

    assert len(result) == 1

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 2
    assert len(edges) >= 1


@pytest.mark.asyncio
async def test_add_data_points_with_triplet_embeddings(clean_test_environment):
    """Integration test: add data points with triplet embeddings enabled."""
    person1 = Person(name="Frank", age=40)
    person2 = Person(name="Grace", age=38)

    custom_edge = (str(person1.id), str(person2.id), "married_to", {"edge_text": "is married to"})

    result = await add_data_points(
        [person1, person2], custom_edges=[custom_edge], embed_triplets=True
    )

    assert len(result) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    assert len(nodes) >= 2
    assert len(edges) >= 1


@pytest.mark.asyncio
async def test_add_data_points_mixed_types(clean_test_environment):
    """Integration test: add different types of DataPoints together."""
    person = Person(name="Ivy", age=28)
    company = Company(name="StartupCo", industry="Software")
    document = Document(title="Meeting Notes", content="Discussed quarterly goals")

    result = await add_data_points([person, company, document])

    assert len(result) == 3

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 3


@pytest.mark.asyncio
async def test_add_data_points_validation_not_list(clean_test_environment):
    """Integration test: verify validation for non-list input."""
    person = Person(name="Jack", age=50)

    with pytest.raises(InvalidDataPointsInAddDataPointsError, match="must be a list"):
        await add_data_points(person)


@pytest.mark.asyncio
async def test_add_data_points_validation_not_datapoint(clean_test_environment):
    """Integration test: verify validation for non-DataPoint items."""
    with pytest.raises(InvalidDataPointsInAddDataPointsError, match="must be a DataPoint"):
        await add_data_points(["not", "datapoints"])


@pytest.mark.asyncio
async def test_add_data_points_with_complex_relationships(clean_test_environment):
    """Integration test: complex graph with multiple relationships."""

    class Project(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class Task(DataPoint):
        title: str
        belongs_to: Project
        assigned_to: Person
        metadata: dict = {"index_fields": ["title"]}

    person = Person(name="Karen", age=35)
    project = Project(name="AI Initiative")
    task = Task(title="Implement feature", belongs_to=project, assigned_to=person)

    result = await add_data_points([task])

    assert len(result) == 1

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    assert len(nodes) >= 3
    assert len(edges) >= 2


@pytest.mark.asyncio
async def test_add_data_points_multiple_batches(clean_test_environment):
    """Integration test: add data points in multiple batches."""
    batch1 = [Person(name="Leo", age=25), Person(name="Mia", age=30)]
    batch2 = [Person(name="Noah", age=35), Person(name="Olivia", age=40)]

    result1 = await add_data_points(batch1)
    result2 = await add_data_points(batch2)

    assert len(result1) == 2
    assert len(result2) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 4


@pytest.mark.asyncio
async def test_add_data_points_with_bidirectional_edges(clean_test_environment):
    """Integration test: add bidirectional relationships."""
    person1 = Person(name="Paul", age=33)
    person2 = Person(name="Quinn", age=31)

    edge1 = (str(person1.id), str(person2.id), "colleague_of", {"edge_text": "works with"})
    edge2 = (str(person2.id), str(person1.id), "colleague_of", {"edge_text": "works with"})

    result = await add_data_points([person1, person2], custom_edges=[edge1, edge2])

    assert len(result) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(edges) >= 2
