import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.modules.topology.topology import TopologyEngine, GitHubRepositoryModel
from cognee.infrastructure.databases.graph.config import get_graph_config

USER_ID = "default_user"

async def add_topology(directory: str = "example", model: BaseModel = GitHubRepositoryModel) -> Any:
    graph_config = get_graph_config()
    graph_db_type = graph_config.graph_database_provider

    graph_client = await get_graph_client(graph_db_type)

    engine = TopologyEngine()
    topology = await engine.infer_from_directory_structure(node_id=USER_ID, repository=directory, model=model)

    def flatten_model(model: BaseModel, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Flatten a single Pydantic model to a dictionary handling nested structures."""
        result = {**model.dict(), "parent_id": parent_id}
        if hasattr(model, "default_relationship") and model.default_relationship:
            result.update({
                "relationship_type": model.default_relationship.type,
                "relationship_source": model.default_relationship.source,
                "relationship_target": model.default_relationship.target
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
                    flat.extend(recursive_flatten(value, items.dict().get("node_id", None)))
            return flat
        else:
            return []

    def flatten_repository(repo_model: BaseModel) -> List[Dict[str, Any]]:
        """ Flatten the entire repository model, starting with the top-level model """
        return recursive_flatten(repo_model)

    flt_topology = flatten_repository(topology)

    df = pd.DataFrame(flt_topology)

    print(df.head(10))

    for _, row in df.iterrows():
        node_data = row.to_dict()
        node_id = node_data.pop("node_id")

        # Remove "node_id" and get its value
        await graph_client.add_node(node_id, node_data)
        if pd.notna(row["relationship_source"]) and pd.notna(row["relationship_target"]):
            await graph_client.add_edge(row["relationship_source"], row["relationship_target"], relationship_name=row["relationship_type"])

    return graph_client.graph
