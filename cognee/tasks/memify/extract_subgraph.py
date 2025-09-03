from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph


async def extract_subgraph(subgraphs: list[CogneeGraph]):
    for subgraph in subgraphs:
        for edge in subgraph.edges:
            yield edge
