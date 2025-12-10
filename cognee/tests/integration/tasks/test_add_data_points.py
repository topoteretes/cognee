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
async def test_add_data_points_comprehensive(clean_test_environment):
    """Comprehensive integration test for add_data_points functionality."""

    person1 = Person(name="Alice", age=30)
    person2 = Person(name="Bob", age=25)
    result = await add_data_points([person1, person2])

    assert result == [person1, person2]
    assert len(result) == 2

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 2

    result_empty = await add_data_points([])
    assert result_empty == []

    person3 = Person(name="Charlie", age=35)
    person4 = Person(name="Diana", age=32)
    custom_edge = (str(person3.id), str(person4.id), "knows", {"edge_text": "friends with"})

    result_custom = await add_data_points([person3, person4], custom_edges=[custom_edge])
    assert len(result_custom) == 2

    nodes, edges = await graph_engine.get_graph_data()
    assert len(edges) == 1
    assert len(nodes) == 4

    class Employee(DataPoint):
        name: str
        works_at: Company
        metadata: dict = {"index_fields": ["name"]}

    company = Company(name="TechCorp", industry="Technology")
    employee = Employee(name="Eve", works_at=company)

    result_rel = await add_data_points([employee])
    assert len(result_rel) == 1

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 6
    assert len(edges) == 2

    person5 = Person(name="Frank", age=40)
    person6 = Person(name="Grace", age=38)
    triplet_edge = (str(person5.id), str(person6.id), "married_to", {"edge_text": "is married to"})

    result_triplet = await add_data_points(
        [person5, person6], custom_edges=[triplet_edge], embed_triplets=True
    )
    assert len(result_triplet) == 2

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 8
    assert len(edges) == 3

    batch1 = [Person(name="Leo", age=25), Person(name="Mia", age=30)]
    batch2 = [Person(name="Noah", age=35), Person(name="Olivia", age=40)]

    result_batch1 = await add_data_points(batch1)
    result_batch2 = await add_data_points(batch2)

    assert len(result_batch1) == 2
    assert len(result_batch2) == 2

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 12
    assert len(edges) == 3

    person7 = Person(name="Paul", age=33)
    person8 = Person(name="Quinn", age=31)
    edge1 = (str(person7.id), str(person8.id), "colleague_of", {"edge_text": "works with"})
    edge2 = (str(person8.id), str(person7.id), "colleague_of", {"edge_text": "works with"})

    result_bi = await add_data_points([person7, person8], custom_edges=[edge1, edge2])
    assert len(result_bi) == 2

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 14
    assert len(edges) == 5

    person_invalid = Person(name="Invalid", age=50)
    with pytest.raises(InvalidDataPointsInAddDataPointsError, match="must be a list"):
        await add_data_points(person_invalid)

    with pytest.raises(InvalidDataPointsInAddDataPointsError, match="must be a DataPoint"):
        await add_data_points(["not", "datapoints"])

    final_nodes, final_edges = await graph_engine.get_graph_data()
    assert len(final_nodes) == 14
    assert len(final_edges) == 5
