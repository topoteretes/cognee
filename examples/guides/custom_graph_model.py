import asyncio
import os
from typing import Annotated, Literal

from cognee import forget, remember, visualize_graph
from cognee.infrastructure.engine import Edge, FromIdentity
from cognee.low_level import DataPoint

CUSTOM_PROMPT = (
    "Extract every person, the role they hold, and every group with its members. "
    "Extract friendships, family links (married_to or sibling_of), who reports to whom, "
    "and other named relationships between people."
)


class Role(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    is_a: Annotated[Role, FromIdentity()] | None = None
    # An edge can also live on the node that owns it. Endpoints of the same type have to
    # be named as strings here, because Person is not bound inside its own body yet.
    # Put an edge here when one side clearly owns it, as each person has one manager.
    reports_to: list[Edge["Person", "Person"]] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Group(DataPoint):
    name: str
    members: list[Person] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class PeopleGraph(DataPoint):
    # Edges on the root suit a relationship with no obvious owner. Each one shows a way
    # of naming: fixed by the field, chosen from a Literal, or free-form from the LLM.
    people: list[Person]
    groups: list[Group] = []
    friends_with: list[Edge[Person, Person]] = []
    family_links: list[Edge[Person, Person, Literal["married_to", "sibling_of"]]] = []
    other_links: list[Edge[Person, Person, str]] = []


async def main():
    await forget(everything=True)

    text = (
        "Maya and Owen are engineers on the Search team and are friends. "
        "Priya is a manager and Maya's sibling. Owen mentors Maya. "
        "Maya and Owen both report to Priya."
    )

    await remember(
        text,
        graph_model=PeopleGraph,
        custom_prompt=CUSTOM_PROMPT,
        self_improvement=False,
    )

    graph_path = os.path.join(os.path.dirname(__file__), ".artifacts", "custom_graph.html")
    await visualize_graph(graph_path)


if __name__ == "__main__":
    asyncio.run(main())
