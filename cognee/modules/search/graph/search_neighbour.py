""" Fetches the context of a given node in the graph"""
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
async def search_neighbour(graph, id,other_param:dict = None):

    node_id = other_param.get('node_id') if other_param else None

    if node_id is None or node_id not in graph:
        return {}

    relevant_context = []
    for n,attr in graph.nodes(data=True):
        if id in n:
            for n_, attr_ in graph.nodes(data=True):
                relevant_layer = attr['layer_uuid']

                if attr_.get('layer_uuid') == relevant_layer:
                    relevant_context.append(attr_['description'])

    return relevant_context



# if __name__ == '__main__':
#     import asyncio
#     async def main():
#         from cognee.shared.data_models import GraphDBType
#
#         graph_client = get_graph_client(GraphDBType.NETWORKX)
#         graph = await  graph_client.graph
#
#         await fetch_context(graph, "1")
#
#     asyncio.run(main())


