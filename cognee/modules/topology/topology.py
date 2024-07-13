""" This module contains the TopologyEngine class which is responsible for adding graph topology from a JSON or CSV file. """

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Type

import asyncio
import aiofiles
import pandas as pd
from pydantic import BaseModel

from cognee.infrastructure.data.chunking.config import get_chunk_config
from cognee.infrastructure.data.chunking.get_chunking_engine import get_chunk_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relationaldb_config
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type, FileTypeException
from cognee.modules.cognify.config import get_cognify_config
from cognee.base_config import get_base_config
from cognee.modules.topology.topology_data_models import NodeModel

cognify_config = get_cognify_config()
base_config = get_base_config()

logger = logging.getLogger("topology")

class TopologyEngine:
    def __init__(self, infer:bool) -> None:
        self.models: Dict[str, Type[BaseModel]] = {}
        self.infer = infer

    async def flatten_model(self, model: NodeModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Flatten the model to a dictionary."""
        result = model.dict()
        result["parent_id"] = parent_id
        if model.default_relationship:
            result.update({
                "relationship_type": model.default_relationship.type,
                "relationship_source": model.default_relationship.source,
                "relationship_target": model.default_relationship.target
            })
        return result

    async def recursive_flatten(self, items: Union[List[Dict[str, Any]], Dict[str, Any]], parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recursively flatten the items.  """
        flat_list = []

        if isinstance(items, list):
            for item in items:
                flat_list.extend(await self.recursive_flatten(item, parent_id))
        elif isinstance(items, dict):
            model = NodeModel.parse_obj(items)
            flat_list.append(await self.flatten_model(model, parent_id))
            for child in model.children:
                flat_list.extend(await self.recursive_flatten(child, model.node_id))
        return flat_list

    async def load_data(self, file_path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data from a JSON or CSV file."""
        try:
            if file_path.endswith(".json"):
                async with aiofiles.open(file_path, mode="r") as f:
                    data = await f.read()
                    return json.loads(data)
            elif file_path.endswith(".csv"):
                async with aiofiles.open(file_path, mode="r") as f:
                    content = await f.read()
                    reader = csv.DictReader(content.splitlines())
                    return list(reader)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {e}")

    async def add_graph_topology(self, file_path: str=None, dataset_files: list[tuple[Any, Any]]=None,):
        """Add graph topology from a JSON or CSV file."""
        if self.infer:
            from cognee.modules.topology.infer_data_topology import infer_data_topology

            initial_chunks_and_ids = []

            chunk_config = get_chunk_config()
            chunk_engine = get_chunk_engine()
            chunk_strategy = chunk_config.chunk_strategy



            for dataset_name, files in dataset_files:
                for base_file in files:
                    with open(base_file["file_path"], "rb") as file:
                        try:
                            file_type = guess_file_type(file)
                            text = extract_text_from_file(file, file_type)

                            subchunks, chunks_with_ids = chunk_engine.chunk_data(chunk_strategy, text, chunk_config.chunk_size,
                                                                   chunk_config.chunk_overlap)

                            if chunks_with_ids[0][0] == 1:
                                initial_chunks_and_ids.append({base_file["id"]: chunks_with_ids})

                        except FileTypeException:
                            logger.warning("File (%s) has an unknown file type. We are skipping it.", file["id"])


            topology = await infer_data_topology(str(initial_chunks_and_ids))
            graph_client = await get_graph_engine()

            for node in topology["nodes"]:
                await graph_client.add_node(node["id"], node)
            for edge in topology["edges"]:
                await graph_client.add_edge(edge["source_node_id"], edge["target_node_id"], relationship_name=edge["relationship_name"])




        else:
            dataset_level_information = dataset_files[0][1]

            # Extract the list of valid IDs from the explanations
            valid_ids = {item["id"] for item in dataset_level_information}
            try:
                data = await self.load_data(file_path)
                flt_topology = await self.recursive_flatten(data)
                df = pd.DataFrame(flt_topology)
                graph_client = await get_graph_engine()

                for _, row in df.iterrows():
                    node_data = row.to_dict()
                    node_id = node_data.pop("node_id", None)
                    if node_id in valid_ids:
                        await graph_client.add_node(node_id, node_data)
                    if node_id not in valid_ids:
                        raise ValueError(f"Node ID {node_id} not found in the dataset")
                    if pd.notna(row.get("relationship_source")) and pd.notna(row.get("relationship_target")):
                        await graph_client.add_edge(row["relationship_source"], row["relationship_target"], relationship_name=row["relationship_type"])

                for dataset in dataset_files:
                    print("dataset", dataset)

                return
            except Exception as e:
                raise RuntimeError(f"Failed to add graph topology from {file_path}: {e}") from e



async def main():
    # text = """Conservative PP in the lead in Spain, according to estimate
    #         An estimate has been published for Spain:
    #
    #         Opposition leader Alberto Núñez Feijóo’s conservative People’s party (PP): 32.4%
    #
    #         Spanish prime minister Pedro Sánchez’s Socialist party (PSOE): 30.2%
    #
    #         The far-right Vox party: 10.4%
    #
    #         In Spain, the right has sought to turn the European election into a referendum on Sánchez.
    #
    #         Ahead of the vote, public attention has focused on a saga embroiling the prime minister’s wife, Begoña Gómez, who is being investigated over allegations of corruption and influence-peddling, which Sanchez has dismissed as politically-motivated and totally baseless."""
    # text_two = """The far-right Vox party: 10.4%"""

    from cognee.api.v1.add import add
    dataset_name = "explanations"
    print(os.getcwd())
    data_dir = os.path.abspath("../../../.data")
    print(os.getcwd())

    await add(f"data://{data_dir}", dataset_name="explanations")

    relational_config = get_relationaldb_config()
    db_engine = relational_config.database_engine


    datasets = db_engine.get_datasets()
    dataset_files =[]

    for added_dataset in datasets:
        if dataset_name in added_dataset:
            dataset_files.append((added_dataset, db_engine.get_files_metadata(added_dataset)))



    print(dataset_files)
    topology_engine = TopologyEngine(infer=True)
    file_path = "example_data.json"  # or 'example_data.csv'
    #
    # # Adding graph topology
    graph = await topology_engine.add_graph_topology(file_path, dataset_files=dataset_files)
    print(graph)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
