"""Define custom DataPoint subclasses and store them with add_data_points, no LLM extraction.

Person nodes are linked through a ``knows`` field as a bare reference, a list, or an
(Edge, target) tuple carrying a weight and a custom relationship_type. Nothing is printed; inspect
the graph afterwards.

Run: uv run python examples/guides/custom_data_models.py
"""

import asyncio
from typing import Any

from pydantic import SkipValidation

import cognee
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.tasks.storage import add_data_points


class Person(DataPoint):
    name: str
    # Keep it simple for forward refs / mixed values
    knows: SkipValidation[Any] = None  # single Person or list[Person]
    # Recommended: specify which fields to index for search
    metadata: dict = {"index_fields": ["name"]}


async def main():
    # Start clean (optional in your app)
    await cognee.forget(everything=True)

    alice = Person(name="Alice")
    bob = Person(name="Bob")
    charlie = Person(name="Charlie")

    # Create relationships - field name becomes edge label
    alice.knows = bob
    # You can also do lists: alice.knows = [bob, charlie]

    # Optional: add weights and custom relationship types
    bob.knows = (Edge(weight=0.9, relationship_type="friend_of"), charlie)

    await add_data_points([alice, bob, charlie])


if __name__ == "__main__":
    asyncio.run(main())
