

from cognee.modules.pipelines.tasks.Task import Task

from cognee.modules.data.extraction.knowledge_graph.establish_graph_topology import establish_graph_topology
from cognee.shared.data_models import KnowledgeGraph



async def ontology_task():
    return Task(establish_graph_topology, topology_model = KnowledgeGraph, task_config = { "batch_size": 10 })