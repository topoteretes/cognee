import asyncio
import os
from typing import List

from cognee import forget, remember, visualize_graph
from cognee.low_level import DataPoint

CUSTOM_PROMPT = (
    "Extract all people mentioned in the text. "
    "For each person, extract ALL activities they like, including shared activities."
)


class Activity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    likes: List[Activity] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class PeopleGraph(DataPoint):
    people: List[Person]


async def main():
    await forget(everything=True)

    text = (
        "Alice likes biking and swimming. Bob likes playing basketball. "
        "Alice and Bob are friends. "
        "Charlie likes skiing. "
        "Alice and Bob like playing board games together."
    )

    await remember(
        text,
        graph_model=PeopleGraph,
        custom_prompt=CUSTOM_PROMPT,
        self_improvement=False,
    )

    graph_path = os.path.join(os.path.dirname(__file__), ".artifacts", "hobbies_graph.html")
    await visualize_graph(graph_path)


if __name__ == "__main__":
    asyncio.run(main())
