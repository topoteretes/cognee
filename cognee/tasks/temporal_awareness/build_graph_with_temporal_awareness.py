import os
from typing import List
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from cognee.infrastructure.files.storage import get_file_storage
from cognee.modules.data.models import Data


async def build_graph_with_temporal_awareness(data: List[Data]):
    text_list: List[str] = []

    for text_data in data:
        file_dir = os.path.dirname(text_data.raw_data_location)
        file_name = os.path.basename(text_data.raw_data_location)
        file_storage = get_file_storage(file_dir)
        async with file_storage.open(file_name, "r") as file:
            text_list.append(file.read())

    url = os.getenv("GRAPH_DATABASE_URL", "")
    password = os.getenv("GRAPH_DATABASE_PASSWORD", "")
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
