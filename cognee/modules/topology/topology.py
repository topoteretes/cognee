# import csv
# import json
# import os
# import glob
#
# import aiofiles
# from pydantic import BaseModel, Field
# from typing import Dict, List, Optional, Union, Type, Any, Tuple
#
# from cognee import config
# from cognee.base_config import get_base_config
# from cognee.infrastructure.databases.graph import get_graph_config
# from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
# from cognee.modules.cognify.config import get_cognify_config
# import pandas as pd
# from pydantic import BaseModel, Field
# from typing import Any, Dict, List, Optional, Union
#
#
# class RelationshipModel(BaseModel):
#     type: str
#     source: str
#     target: str
#
#
# class NodeModel(BaseModel):
#     node_id: str
#     name: str
#     default_relationship: Optional[RelationshipModel] = None
#     children: List[Union[Dict[str, Any], "NodeModel"]] = Field(default_factory=list)
#
#
# NodeModel.update_forward_refs()
# cognify_config = get_cognify_config()
# base_config = get_base_config()
#
# class Relationship(BaseModel):
#     type: str = Field(..., description="The type of relationship, e.g., 'belongs_to'.")
#     source: Optional[str] = Field(None, description="The identifier of the source id of in the relationship being a directory or subdirectory")
#     target: Optional[str] = Field(None, description="The identifier of the target id in the relationship being the directory, subdirectory or file")
#     properties: Optional[Dict[str, Any]] = Field(None, description="A dictionary of additional properties and values related to the relationship.")
#
#
# class Document(BaseModel):
#     node_id: str
#     title: str
#     description: Optional[str] = None
#     default_relationship: Relationship
#
#
# class DirectoryModel(BaseModel):
#     node_id: str
#     path: str
#     summary: str
#     documents: List[Document] = []
#     subdirectories: List['DirectoryModel'] = []
#     default_relationship: Relationship
#
#
# DirectoryModel.update_forward_refs()
#
#
# class DirMetadata(BaseModel):
#     node_id: str
#     summary: str
#     owner: str
#     description: Optional[str] = None
#     directories: List[DirectoryModel] = []
#     documents: List[Document] = []
#     default_relationship: Relationship
#
#
# class GitHubRepositoryModel(BaseModel):
#     node_id: str
#     metadata: DirMetadata
#     root_directory: DirectoryModel
#
#
# class TopologyEngine:
#     def __init__(self) -> None:
#         self.models: Dict[str, Type[BaseModel]] = {}
#         self.infer = False
#     async def flatten_model(self, model: NodeModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
#         result = model.dict()
#         result["parent_id"] = parent_id
#         if model.default_relationship:
#             result.update({
#                 "relationship_type": model.default_relationship.type,
#                 "relationship_source": model.default_relationship.source,
#                 "relationship_target": model.default_relationship.target
#             })
#         return result
#
#     async def recursive_flatten(self, items: Union[List[Dict[str, Any]], Dict[str, Any]], parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
#         flat_list = []
#
#         if isinstance(items, list):
#             for item in items:
#                 flat_list.extend(await self.recursive_flatten(item, parent_id))
#         elif isinstance(items, dict):
#             item = NodeModel.parse_obj(items)
#             flat_list.append(await self.flatten_model(item, parent_id))
#             for child in item.children:
#                 flat_list.extend(await self.recursive_flatten(child, item.node_id))
#         return flat_list
#
#     async def load_data(self, file_path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
#         if file_path.endswith('.json'):
#             async with aiofiles.open(file_path, mode='r') as f:
#                 data = await f.read()
#                 return json.loads(data)
#         elif file_path.endswith('.csv'):
#             async with aiofiles.open(file_path, mode='r') as f:
#                 reader = csv.DictReader(await f.read().splitlines())
#                 return list(reader)
#         else:
#             raise ValueError("Unsupported file format")
#
#     async def add_graph_topology(self, file_path: str):
#         data = await self.load_data(file_path)
#
#         flt_topology = await self.recursive_flatten(data)
#         print(flt_topology)
#         df = pd.DataFrame(flt_topology)
#         graph_client = await get_graph_client()
#
#         for _, row in df.iterrows():
#             node_data = row.to_dict()
#             node_id = node_data.pop("node_id", None)
#             await graph_client.add_node(node_id, node_data)
#             if pd.notna(row["relationship_source"]) and pd.notna(row["relationship_target"]):
#                 await graph_client.add_edge(row["relationship_source"], row["relationship_target"], relationship_name=row["relationship_type"])
#
#         return graph_client.graph
#
#
#
#
# if __name__ == "__main__":
#     async def main():
#         topology_engine = TopologyEngine()
#         file_path = 'example_data.json'  # or 'example_data.csv'
#
#         # Adding graph topology
#         graph = await topology_engine.add_graph_topology(file_path)
#         print(graph)
#
#
#     import asyncio
#     asyncio.run(main())
#     # result = engine.extrapolate("GitHubRepositoryModel")
#     # print(result)

import csv
import json
import aiofiles
import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Type

from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.modules.cognify.config import get_cognify_config
from cognee.base_config import get_base_config

# Define models
class RelationshipModel(BaseModel):
    type: str
    source: str
    target: str

class NodeModel(BaseModel):
    node_id: str
    name: str
    default_relationship: Optional[RelationshipModel] = None
    children: List[Union[Dict[str, Any], "NodeModel"]] = Field(default_factory=list)

NodeModel.update_forward_refs()
cognify_config = get_cognify_config()
base_config = get_base_config()

class Relationship(BaseModel):
    type: str = Field(..., description="The type of relationship, e.g., 'belongs_to'.")
    source: Optional[str] = Field(None, description="The identifier of the source id in the relationship.")
    target: Optional[str] = Field(None, description="The identifier of the target id in the relationship.")
    properties: Optional[Dict[str, Any]] = Field(None, description="A dictionary of additional properties related to the relationship.")

class Document(BaseModel):
    node_id: str
    title: str
    description: Optional[str] = None
    default_relationship: Relationship

class DirectoryModel(BaseModel):
    node_id: str
    path: str
    summary: str
    documents: List[Document] = []
    subdirectories: List['DirectoryModel'] = []
    default_relationship: Relationship

DirectoryModel.update_forward_refs()

class DirMetadata(BaseModel):
    node_id: str
    summary: str
    owner: str
    description: Optional[str] = None
    directories: List[DirectoryModel] = []
    documents: List[Document] = []
    default_relationship: Relationship

class GitHubRepositoryModel(BaseModel):
    node_id: str
    metadata: DirMetadata
    root_directory: DirectoryModel

class TopologyEngine:
    def __init__(self) -> None:
        self.models: Dict[str, Type[BaseModel]] = {}
        self.infer = False

    async def flatten_model(self, model: NodeModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
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

# Example Main Function:
import asyncio

async def main():
    topology_engine = TopologyEngine()
    file_path = 'example_data.json'  # or 'example_data.csv'

    # Adding graph topology
    graph = await topology_engine.add_graph_topology(file_path)
    print(graph)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
