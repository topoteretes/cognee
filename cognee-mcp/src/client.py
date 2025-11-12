from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python",
    args=[
        "-m",
        "cognee-mcp.src.server",
        "--transport",
        "stdio",
        "--api-url",
        "http://localhost:8000",
    ],
)


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, timedelta(minutes=1)) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools_result.tools])

            datasets = await session.call_tool("list_datasets", arguments={})
            print("Datasets response:", datasets.content)

            search_arguments = {
                "query": "Give me a short summary of the onboarding guide",
                "search_type": "GRAPH_COMPLETION",
                "top_k": 5,
            }
            search_result = await session.call_tool("search", arguments=search_arguments)
            print("Search response:", search_result.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
