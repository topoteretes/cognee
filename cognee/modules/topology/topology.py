""" This module contains the TopologyEngine class which is responsible for adding graph topology from a JSON or CSV file. """

import csv
import json
import aiofiles
import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Type
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.modules.cognify.config import get_cognify_config
from cognee.base_config import get_base_config
from cognee.modules.topology.topology_data_models import NodeModel, RelationshipModel, Document, DirectoryModel, DirMetadata, GitHubRepositoryModel
import asyncio
cognify_config = get_cognify_config()
base_config = get_base_config()

class TopologyEngine:
    def __init__(self) -> None:
        self.models: Dict[str, Type[BaseModel]] = {}
        self.infer = False

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
            if file_path.endswith('.json'):
                async with aiofiles.open(file_path, mode='r') as f:
                    data = await f.read()
                    return json.loads(data)
            elif file_path.endswith('.csv'):
                async with aiofiles.open(file_path, mode='r') as f:
                    content = await f.read()
                    reader = csv.DictReader(content.splitlines())
                    return list(reader)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {e}")

    async def add_graph_topology(self, file_path: str):
        """Add graph topology from a JSON or CSV file."""
        try:
            data = await self.load_data(file_path)
            flt_topology = await self.recursive_flatten(data)
            df = pd.DataFrame(flt_topology)
            graph_client = await get_graph_client()

            for _, row in df.iterrows():
                node_data = row.to_dict()
                node_id = node_data.pop("node_id", None)
                await graph_client.add_node(node_id, node_data)
                if pd.notna(row.get("relationship_source")) and pd.notna(row.get("relationship_target")):
                    await graph_client.add_edge(row["relationship_source"], row["relationship_target"], relationship_name=row["relationship_type"])

            return graph_client.graph
        except Exception as e:
            raise RuntimeError(f"Failed to add graph topology from {file_path}: {e}")



async def main():
    topology_engine = TopologyEngine()
    file_path = 'example_data.json'  # or 'example_data.csv'

    # Adding graph topology
    graph = await topology_engine.add_graph_topology(file_path)
    print(graph)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
