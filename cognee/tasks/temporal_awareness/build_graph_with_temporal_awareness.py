import os
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType


async def build_graph_with_temporal_awareness(text_list):
    url = os.getenv("GRAPH_DATABASE_URL")
    password = os.getenv("GRAPH_DATABASE_PASSWORD")
    graphiti = Graphiti(url, "neo4j", password)

    await graphiti.build_indices_and_constraints()
    print("Graph database initialized.")

    for i, text in enumerate(text_list):
        await graphiti.add_episode(
            name=f"episode_{i}",
            episode_body=text,
            source=EpisodeType.text,
            source_description="input",
            reference_time=datetime.now(),
        )
        print(f"Added text: {text[:35]}...")
    return graphiti
