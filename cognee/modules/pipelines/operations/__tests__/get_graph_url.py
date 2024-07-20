import asyncio
from cognee.shared.utils import render_graph
from cognee.infrastructure.databases.graph import get_graph_engine

if __name__ == "__main__":
    async def main():
        graph_client = await get_graph_engine()
        graph = graph_client.graph

        graph_url = await render_graph(graph)

        print(graph_url)

    asyncio.run(main())
