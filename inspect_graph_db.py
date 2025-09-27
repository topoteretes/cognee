import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine

async def print_graph():
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    print("Nodes:")
    for node in nodes:
        print(node)
    print("\nEdges:")
    for edge in edges:
        print(edge)

if __name__ == "__main__":
    asyncio.run(print_graph())
