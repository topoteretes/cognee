import asyncio
import os
from typing import Annotated, Literal

from cognee import forget, remember, visualize_graph
from cognee.infrastructure.engine import Edge, FromIdentity
from cognee.low_level import DataPoint

CUSTOM_PROMPT = (
    "Extract every person, the role they hold, and every group with its members. "
    "Extract friendships, family links (married_to or sibling_of), "
    "and other named relationships between people."
)


class Role(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    is_a: Annotated[Role, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Group(DataPoint):
    name: str
    members: list[Person] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class PeopleGraph(DataPoint):
    people: list[Person]
    groups: list[Group] = []
    friends_with: list[Edge[Person, Person]] = []
    family_links: list[Edge[Person, Person, Literal["married_to", "sibling_of"]]] = []
    other_links: list[Edge[Person, Person, str]] = []


async def main():
    await forget(everything=True)

    text = (
        "Maya and Owen are engineers on the Search team and are friends. "
        "Priya is a manager and Maya's sibling. Owen mentors Maya."
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
