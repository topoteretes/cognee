import asyncio
from cognee.shared.utils import render_graph
from cognee.infrastructure.databases.graph import get_graph_engine

if __name__ == "__main__":

    async def main():
        import os
        import pathlib
        import cognee

        data_directory_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_library")
            ).resolve()
        )
        cognee.config.data_root_directory(data_directory_path)
        cognee_directory_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_library")
            ).resolve()
        )
        cognee.config.system_root_directory(cognee_directory_path)

        graph_client = await get_graph_engine()
        graph = graph_client.graph

        graph_url = await render_graph(graph)

        print(graph_url)

    asyncio.run(main())
