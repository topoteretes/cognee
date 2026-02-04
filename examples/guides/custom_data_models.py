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
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

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
