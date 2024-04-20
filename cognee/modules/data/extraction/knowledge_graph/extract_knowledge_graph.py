import asyncio
import logging
from cognee.root_dir import get_absolute_path
from .extract_knowledge_graph_module import ExtractKnowledgeGraph
from .extract_content_graph import extract_content_graph

logger = logging.getLogger("extract_knowledge_graph(text: str)")

async def extract_knowledge_graph(text: str, cognitive_layer, graph_model):
    try:
        compiled_extract_knowledge_graph = ExtractKnowledgeGraph()
        compiled_extract_knowledge_graph.load(get_absolute_path("./programs/extract_knowledge_graph/extract_knowledge_graph.json"))

        event_loop = asyncio.get_event_loop()

        def sync_extract_knowledge_graph():
            return compiled_extract_knowledge_graph(context = text, question = "")

        return (await event_loop.run_in_executor(None, sync_extract_knowledge_graph)).graph
        # return compiled_extract_knowledge_graph(text, question = "").graph
    except Exception as error:
        logger.error("Error extracting graph from content: %s", error, exc_info = True)
        
        return await extract_content_graph(text, cognitive_layer, graph_model)
