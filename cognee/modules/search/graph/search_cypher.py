
from typing import Union, Dict
import re

import networkx as nx
from pydantic import BaseModel

from cognee.modules.search.llm.extraction.categorize_relevant_category import categorize_relevant_category
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure.databases.graph.config import get_graph_config
graph_config = get_graph_config()
from cognee.infrastructure.databases.vector.config import get_vectordb_config
vector_config = get_vectordb_config()

async def search_cypher(query:str, graph: Union[nx.Graph, any]):
    """
    Use a Cypher query to search the graph and return the results.
    """


    from cognee.infrastructure import infrastructure_config
    if graph_config.graph_engine == GraphDBType.NEO4J:
        result = await graph.run(query)
        return result

    else:
        raise ValueError("Unsupported graph engine type.")