import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.engine.models import Entity, EntityType

restore_module = importlib.import_module(
    "cognee.modules.graph.utils.restore_entity_type_categories"
)


def _entity_type(name: str, category: str | None = None) -> EntityType:
    return EntityType(name=name, description=name, category=category)


def _by_id(*data_points) -> dict:
    return {str(data_point.id): data_point for data_point in data_points}


def _graph_engine(stored_nodes: list[dict]) -> MagicMock:
    engine = MagicMock()
    engine.get_nodes = AsyncMock(return_value=stored_nodes)
    return engine


@pytest.fixture
def stored_nodes():
    """Patch the graph engine the helper reads from; the test sets what it holds."""
    engine = _graph_engine([])
    with patch.object(restore_module, "get_graph_engine", AsyncMock(return_value=engine)):
        yield engine


@pytest.mark.asyncio
async def test_stored_category_is_copied_onto_the_new_instance(stored_nodes):
    person = _entity_type("person")
    country = _entity_type("country")
    stored_nodes.get_nodes.return_value = [{"id": str(person.id), "category": "person"}]

    await restore_module.restore_entity_type_categories(_by_id(person, country))

    assert person.category == "person"
    assert country.category is None


@pytest.mark.asyncio
async def test_only_unclassified_entity_types_are_read(stored_nodes):
    """A category set this run is not looked up, and an Entity is not an EntityType."""
    classified = _entity_type("place", category="place")
    unclassified = _entity_type("person")
    entity = Entity(name="alice", description="alice", is_a=unclassified)

    await restore_module.restore_entity_type_categories(_by_id(classified, unclassified, entity))

    stored_nodes.get_nodes.assert_awaited_once_with([str(unclassified.id)])


@pytest.mark.asyncio
async def test_nothing_unclassified_makes_no_graph_call():
    with patch.object(restore_module, "get_graph_engine", AsyncMock()) as get_engine:
        await restore_module.restore_entity_type_categories(
            _by_id(_entity_type("place", category="place"))
        )

    get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stored_node_without_a_category_leaves_it_unclassified(stored_nodes):
    """Nodes written before the field existed carry no key, and a null is the same thing."""
    person = _entity_type("person")
    country = _entity_type("country")
    stored_nodes.get_nodes.return_value = [
        {"id": str(person.id)},
        {"id": str(country.id), "category": None},
    ]

    await restore_module.restore_entity_type_categories(_by_id(person, country))

    assert person.category is None and country.category is None
