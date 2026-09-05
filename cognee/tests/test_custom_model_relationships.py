import os
import pathlib
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import Edge
from cognee.low_level import DataPoint


async def main():
    class Activity(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Person(DataPoint):
        name: str
        likes: list[Activity] | None = None
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class PeopleGraph(DataPoint):
        people: list[Person]
        friends_with: list[Edge[Person, Person]]

    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent, ".data_storage/test_custom_model_relationships"
            )
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent, ".cognee_system/test_custom_model_relationships"
            )
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = (
        "Alice likes biking and swimming. Bob likes playing basketball. "
        "Alice and Bob are friends. "
        "Charlie likes skiing. "
        "Alice and Bob like playing board games together."
    )
    custom_prompt = (
        "Extract all people mentioned in the text. "
        "For each person, extract ALL activities they like, including shared activities. "
        "Also extract friendships between people."
    )

    await cognee.add(text)
    await cognee.cognify(graph_model=PeopleGraph, custom_prompt=custom_prompt)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    nodes_by_id = {str(node[0]): node[1] for node in nodes}
    friend_pairs = {
        (nodes_by_id[str(edge[0])].get("name"), nodes_by_id[str(edge[1])].get("name"))
        for edge in edges
        if edge[2] == "friends_with"
    }
    assert friend_pairs, f"No friends_with edges in { {edge[2] for edge in edges} }"
    assert ("Alice", "Bob") in friend_pairs or ("Bob", "Alice") in friend_pairs, friend_pairs


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
