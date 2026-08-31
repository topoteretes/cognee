from typing import Annotated

import pytest

from cognee.infrastructure.engine import DataPoint, FromIdentity
from cognee.modules.graph.utils import get_graph_from_model
from cognee.shared.graph_model_utils import (
    content_graph_to_data_point,
    datapoint_model_to_basemodel,
)


class Role(DataPoint):
    name: str
    rank: str = "member"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    is_a: Annotated[Role, FromIdentity()]
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class PeopleGraph(DataPoint):
    people: list[Person]


class NestedOnly(DataPoint):
    name: str
    child: "NestedOnly | None" = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


NestedOnly.model_rebuild()


@pytest.mark.asyncio
async def test_from_identity_strings_become_nodes_deduped_by_id():
    simplified = datapoint_model_to_basemodel(PeopleGraph, strip_metadata=True)
    content_graph = simplified(
        people=[
            {"name": "Alice", "is_a": "Student"},
            {"name": "Bob", "is_a": "Student"},
        ]
    )

    root = await content_graph_to_data_point(content_graph, PeopleGraph)
    assert root.people[0].is_a.id == Role.id_for("Student")
    assert root.people[1].is_a.id == Role.id_for("Student")

    nodes, _ = await get_graph_from_model(root)
    role_nodes = [node for node in nodes if getattr(node, "name", None) == "Student"]
    assert len(role_nodes) == 1


@pytest.mark.asyncio
async def test_from_identity_keeps_defaulted_extra_fields():
    simplified = datapoint_model_to_basemodel(Person, strip_metadata=True)
    content_graph = simplified(name="Alice", is_a="Student")
    person = await content_graph_to_data_point(content_graph, Person)
    assert person.is_a.rank == "member"


@pytest.mark.asyncio
async def test_model_without_from_identity_matches_model_validate():
    simplified = datapoint_model_to_basemodel(NestedOnly, strip_metadata=True)
    content_graph = simplified(name="root", child={"name": "leaf"})
    dump = content_graph.model_dump()
    via_inverse = await content_graph_to_data_point(content_graph, NestedOnly)
    via_validate = NestedOnly.model_validate(dump)
    assert via_inverse.name == via_validate.name
    assert via_inverse.child.name == via_validate.child.name
