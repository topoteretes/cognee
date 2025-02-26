import os
import logging
import asyncio
from typing import Type, List

from cognee.shared.utils import setup_logging as setup_logging
from cognee.infrastructure.llm import get_max_chunk_tokens as get_max_chunk_tokens
from cognee.modules.data.methods import get_datasets as get_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data as get_dataset_data
from cognee.modules.data.models import Data as Data
from cognee.modules.pipelines import run_tasks as run_tasks
from cognee.modules.pipelines.tasks.Task import Task as Task
from cognee.modules.users.methods import get_default_user as get_default_user
from cognee.tasks.documents import (
    check_permissions_on_documents as check_permissions_on_documents,
    classify_documents as classify_documents,
    extract_chunks_from_documents as extract_chunks_from_documents,
)
from cognee.infrastructure.databases.graph import get_graph_engine as get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk as DocumentChunk
from cognee.modules.data.extraction.knowledge_graph import (
    extract_content_graph as extract_content_graph,
)
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges as expand_with_nodes_and_edges,
    retrieve_existing_edges as retrieve_existing_edges,
)
from cognee.api.v1.prune.prune import prune_data, prune_system
from cognee.api.v1.add.add_v2 import add
from cognee.shared.data_models import KnowledgeGraph as KnowledgeGraph
from cognee.tasks.storage import add_data_points as add_data_points
from cognee.api.v1.visualize import visualize_graph



from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from cognee.infrastructure.engine import DataPoint
from cognee.api.v1.search import SearchType
from cognee.api.v1.search.search_v2 import search
from cognee.tasks.storage.index_graph_edges import index_graph_edges
