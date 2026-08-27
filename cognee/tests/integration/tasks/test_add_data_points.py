import pathlib
import re
import shutil
from typing import Any

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
async def clean_test_environment(request):
    """Set up a clean test environment for add_data_points tests.

    Each test gets its own directories. Two tests cannot share one, because the
    graph database is a file the engine locks exclusively and the previous test's
    engine can still hold that lock: its close is deferred until the last handle
    is released, which may not happen before the next test opens the same path.
    """
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    test_slug = re.sub(r"[^0-9A-Za-z]+", "_", request.node.name)
    system_directory_path = str(
        base_dir / ".cognee_system/test_add_data_points_integration" / test_slug
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_add_data_points_integration" / test_slug
    )

    # Start from nothing on disk: with access control on, ``prune_system`` prunes
    # per-dataset databases only, so a graph written straight through
    # ``add_data_points`` outlives it and the counts below would be read against
    # a previous run's leftovers.
    shutil.rmtree(system_directory_path, ignore_errors=True)
    shutil.rmtree(data_directory_path, ignore_errors=True)

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
    # 4 Persons + Employee + Company = 6
    assert len(nodes) == 6
    assert len(edges) == 2
    assert not any(node[1].get("type") == "EdgeType" for node in nodes)

    person5 = Person(name="Frank", age=40)
    person6 = Person(name="Grace", age=38)
    triplet_edge = (str(person5.id), str(person6.id), "married_to", {"edge_text": "is married to"})

    result_triplet = await add_data_points(
        [person5, person6], custom_edges=[triplet_edge], embed_triplets=True
    )
    assert len(result_triplet) == 2

    nodes, edges = await graph_engine.get_graph_data()
    # 6 + 2 Persons = 8
    assert len(nodes) == 8
    assert len(edges) == 3

    batch1 = [Person(name="Leo", age=25), Person(name="Mia", age=30)]
    batch2 = [Person(name="Noah", age=35), Person(name="Olivia", age=40)]

    result_batch1 = await add_data_points(batch1)
    result_batch2 = await add_data_points(batch2)

    assert len(result_batch1) == 2
    assert len(result_batch2) == 2

    nodes, edges = await graph_engine.get_graph_data()
    # 8 + 4 Persons = 12
    assert len(nodes) == 12
    assert len(edges) == 3

    person7 = Person(name="Paul", age=33)
    person8 = Person(name="Quinn", age=31)
    edge1 = (str(person7.id), str(person8.id), "colleague_of", {"edge_text": "works with"})
    edge2 = (str(person8.id), str(person7.id), "colleague_of", {"edge_text": "works with"})

    result_bi = await add_data_points([person7, person8], custom_edges=[edge1, edge2])
    assert len(result_bi) == 2

    nodes, edges = await graph_engine.get_graph_data()
    # 12 + 2 Persons = 14
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


class TeamWrapper(DataPoint):
    """Container the LLM needs to return one object; not a fact about the domain."""

    members: list[Person]
    metadata: dict = {"index_fields": [], "transparent": True}


@pytest.mark.asyncio
async def test_add_data_points_resolves_transparent_wrapper(clean_test_environment):
    """A transparent top-level wrapper stores its children as top-level nodes."""

    alice = Person(name="Alice", age=30)
    bob = Person(name="Bob", age=25)
    wrapper = TeamWrapper(members=[alice, bob])

    graph_engine = await get_graph_engine()
    baseline_nodes, baseline_edges = await graph_engine.get_graph_data()
    baseline_ids = {str(node[0]) for node in baseline_nodes}

    result = await add_data_points([wrapper], graph_only=True)

    assert result == [wrapper]

    nodes, edges = await graph_engine.get_graph_data()
    added_ids = {str(node[0]) for node in nodes} - baseline_ids

    # Only the effective descendants are persisted; the container is not a node,
    # and it cannot be an edge endpoint because it does not exist.
    assert added_ids == {str(alice.id), str(bob.id)}
    assert all(str(wrapper.id) not in (str(source), str(target)) for source, target, *_ in edges)

    # The graph engine synthesizes a SELF edge per node, so compare real relationships.
    def _relationships(graph_edges):
        return [edge for edge in graph_edges if edge[2] != "SELF"]

    assert _relationships(edges) == _relationships(baseline_edges)


class AttachmentChunk(DataPoint):
    """A DocumentChunk-shaped node, so the real walk mints the contains edges."""

    text: str
    contains: Any = None
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_attachment", [None, "all"])
async def test_chunk_attachment_persists_expected_contains_edges(
    clean_test_environment, chunk_attachment
):
    """Transparent wrapper + attachment, end to end through the graph store."""
    from cognee.modules.graph.utils import collect_stored_data_points
    from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs

    alice = Person(name="Alice", age=30)
    bob = Person(name="Bob", age=25)
    wrapper = TeamWrapper(members=[alice, bob])
    chunk = AttachmentChunk(text="Alice is 30. Bob is 25.")

    await integrate_chunk_graphs(
        [chunk], [wrapper], TeamWrapper, None, chunk_attachment=chunk_attachment
    )
    await add_data_points([chunk], graph_only=True)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    stored_ids = {str(node[0]) for node in nodes}
    assert str(wrapper.id) not in stored_ids

    # Scope to this chunk: setup() seeds nodes that carry contains edges of their own.
    contains = {
        str(target)
        for source, target, name, *_ in edges
        if name == "contains" and str(source) == str(chunk.id)
    }
    if chunk_attachment == "all":
        assert contains == {str(node.id) for node in await collect_stored_data_points(wrapper)}
    else:
        assert contains == {str(alice.id), str(bob.id)}
