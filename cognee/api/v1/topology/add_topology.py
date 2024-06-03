from typing import List, Dict, Any, Union, Optional

from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client

from cognee.modules.topology.topology import TopologyEngine, GitHubRepositoryModel
import pandas as pd
from pydantic import BaseModel
import os
import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Type, Any
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.infrastructure import infrastructure_config




class Relationship(BaseModel):
    type: str = Field(..., description="The type of relationship, e.g., 'belongs_to'.")
    source: Optional[str] = Field(None, description="The identifier of the source id of in the relationship being a directory or subdirectory")
    target: Optional[str] = Field(None, description="The identifier of the target id in the relationship being the directory, subdirectory or file")
    properties: Optional[Dict[str, Any]] = Field(None, description="A dictionary of additional properties and values related to the relationship.")

class JSONEntity(BaseModel):
    name: str
    set_type_as: Optional[str] = None
    property_columns: List[str]
    description: Optional[str] = None

class JSONPattern(BaseModel):
    head: str
    relation: str
    tail: str
    description: Optional[str] = None

class JSONModel(BaseModel):
    node_id: str
    entities: List[JSONEntity]
    patterns: List[JSONPattern]
USER_ID = "default_user"

async def add_topology(directory: str = "example", model: BaseModel = GitHubRepositoryModel) -> Any:
    graph_db_type = infrastructure_config.get_config()["graph_engine"]

    graph_client = await get_graph_client(graph_db_type)

    graph_topology = infrastructure_config.get_config()["graph_topology"]

    engine = TopologyEngine()
    topology = await engine.infer_from_directory_structure(node_id=USER_ID, repository=directory, model=model)

    def flatten_model(model: BaseModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Flatten a single Pydantic model to a dictionary handling nested structures."""
        result = {**model.dict(), 'parent_id': parent_id}
        if hasattr(model, 'default_relationship') and model.default_relationship:
            result.update({
                'relationship_type': model.default_relationship.type,
                'relationship_source': model.default_relationship.source,
                'relationship_target': model.default_relationship.target
            })
        return result

    def recursive_flatten(items: Union[List[Any], BaseModel], parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recursively flatten nested Pydantic models or lists of models."""
        if isinstance(items, list):
            return [entry for item in items for entry in recursive_flatten(item, parent_id)]
        elif isinstance(items, BaseModel):
            flat = [flatten_model(items, parent_id)]
            for field, value in items:
                if isinstance(value, (BaseModel, list)):
                    flat.extend(recursive_flatten(value, items.dict().get('node_id', None)))
            return flat
        else:
            return []

    def flatten_repository(repo_model: BaseModel) -> List[Dict[str, Any]]:
        """ Flatten the entire repository model, starting with the top-level model """
        return recursive_flatten(repo_model)

    async def add_graph_topology():

        flt_topology = flatten_repository(topology)

        df = pd.DataFrame(flt_topology)

        print(df.head(10))

        for _, row in df.iterrows():
            node_data = row.to_dict()
            node_id = node_data.pop('node_id')

            # Remove 'node_id' and get its value
            await graph_client.add_node(node_id, node_data)
            if pd.notna(row['relationship_source']) and pd.notna(row['relationship_target']):
                await graph_client.add_edge(row['relationship_source'], row['relationship_target'], relationship_name=row['relationship_type'])

        return graph_client.graph

    await add_graph_topology()



    def parse_json(self, json_data: str) -> JSONModel:
        data = json.loads(json_data)
        entities = [JSONEntity(**entity) for entity in data.get('entities', [])]
        patterns = [JSONPattern(**pattern) for pattern in data.get('patterns', [])]
        return JSONModel(node_id="json_node", entities=entities, patterns=patterns)

    async def add_json_topology(json_data: str, node_id: str = "json_node", model: Type[BaseModel] = JSONModel) -> Any:
        graph_db_type = infrastructure_config.get_config()["graph_engine"]
        graph_client = await get_graph_client(graph_db_type)
        engine = TopologyEngine()
        topology = await engine.infer_from_json(node_id=node_id, json_data=json_data)

        def flatten_model(model: BaseModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
            result = {**model.dict(), 'parent_id': parent_id}
            if hasattr(model, 'default_relationship') and model.default_relationship:
                result.update({
                    'relationship_type': model.default_relationship.type,
                    'relationship_source': model.default_relationship.source,
                    'relationship_target': model.default_relationship.target
                })
            return result

        def recursive_flatten(items: Union[List[Any], BaseModel], parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
            if isinstance(items, list):
                return [entry for item in items for entry in recursive_flatten(item, parent_id)]
            elif isinstance(items, BaseModel):
                flat = [flatten_model(items, parent_id)]
                for field, value in items:
                    if isinstance(value, (BaseModel, list)):
                        flat.extend(recursive_flatten(value, items.dict().get('node_id', None)))
                return flat
            else:
                return []

        def flatten_json_model(json_model: BaseModel) -> List[Dict[str, Any]]:
            return recursive_flatten(json_model)

        flt_topology = flatten_json_model(topology)
        df = pd.DataFrame(flt_topology)

        print(df.head(10))

        for _, row in df.iterrows():
            node_data = row.to_dict()
            node_id = node_data.pop('node_id')

            await graph_client.add_node(node_id, node_data)
            if pd.notna(row.get('relationship_source')) and pd.notna(row.get('relationship_target')):
                await graph_client.add_edge(row['relationship_source'], row['relationship_target'], relationship_name=row['relationship_type'])

        return graph_client.graph



if __name__ == "__main__":
    async def test() -> None:
        # Uncomment and modify the following lines as needed
        # await prune.prune_system()
        #
        # from cognee.api.v1.add import add
        # data_directory_path = os.path.abspath("../../../.data")
        # # print(data_directory_path)
        # # config.data_root_directory(data_directory_path)
        # # cognee_directory_path = os.path.abspath("../.cognee_system")
        # # config.system_root_directory(cognee_directory_path)
        #
        # await add("data://" + data_directory_path, "example")

        # graph = await add_topology()

        graph_db_type = infrastructure_config.get_config()["graph_engine"]

        graph_client = await get_graph_client(graph_db_type)
        #
        from cognee.utils import render_graph

        await render_graph(graph_client.graph, include_color=True, include_nodes=False, include_size=False)

    import asyncio
    asyncio.run(test())
