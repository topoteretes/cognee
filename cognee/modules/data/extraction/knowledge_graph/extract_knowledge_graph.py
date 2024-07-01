from .extract_content_graph import extract_content_graph

async def extract_knowledge_graph(text: str, cognitive_layer, graph_model):
    return await extract_content_graph(text, cognitive_layer, graph_model)
