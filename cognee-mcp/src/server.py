import asyncio
import json
import os
import cognee
import logging
import importlib.util
from contextlib import redirect_stderr, redirect_stdout

# from PIL import Image as PILImage
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.storage.utils import JSONEncoder

mcp = Server("cognee")

logger = logging.getLogger(__name__)


@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="cognify",
            description="Cognifies text into knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to cognify",
                    },
                    "graph_model_file": {
                        "type": "string",
                        "description": "The path to the graph model file",
                    },
                    "graph_model_name": {
                        "type": "string",
                        "description": "The name of the graph model",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="codify",
            description="Transforms codebase into knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                    },
                },
                "required": ["repo_path"],
            },
        ),
        types.Tool(
            name="search",
            description="Searches for information in knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The query to search for",
                    },
                    "search_type": {
                        "type": "string",
                        "description": "The type of search to perform (e.g., INSIGHTS, CODE)",
                    },
                },
                "required": ["search_query"],
            },
        ),
        types.Tool(
            name="prune",
            description="Prunes knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@mcp.call_tool()
async def call_tools(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                if name == "cognify":
                    await cognify(
                        text=arguments["text"],
                        graph_model_file=arguments.get("graph_model_file", None),
                        graph_model_name=arguments.get("graph_model_name", None),
                    )

                    return [types.TextContent(type="text", text="Ingested")]
                if name == "codify":
                    await codify(arguments.get("repo_path"))

                    return [types.TextContent(type="text", text="Indexed")]
                elif name == "search":
                    search_results = await search(
                        arguments["search_query"], arguments["search_type"]
                    )

                    return [types.TextContent(type="text", text=search_results)]
                elif name == "prune":
                    await prune()

                    return [types.TextContent(type="text", text="Pruned")]
    except Exception as e:
        logger.error(f"Error calling tool '{name}': {str(e)}")
        return [types.TextContent(type="text", text=f"Error calling tool '{name}': {str(e)}")]


async def cognify(text: str, graph_model_file: str = None, graph_model_name: str = None) -> str:
    """Build knowledge graph from the input text"""
    if graph_model_file and graph_model_name:
        graph_model = load_class(graph_model_file, graph_model_name)
    else:
        graph_model = KnowledgeGraph

    await cognee.add(text)

    try:
        asyncio.create_task(cognee.cognify(graph_model=graph_model))
    except Exception as e:
        raise ValueError(f"Failed to cognify: {str(e)}")


async def codify(repo_path: str):
    async for result in run_code_graph_pipeline(repo_path, False):
        logger.info(result)


async def search(search_query: str, search_type: str) -> str:
    """Search the knowledge graph"""
    search_results = await cognee.search(
        query_type=SearchType[search_type.upper()], query_text=search_query
    )

    if search_type.upper() == "CODE":
        return json.dumps(search_results, cls=JSONEncoder)
    else:
        results = retrieved_edges_to_string(search_results)
        return results


async def prune():
    """Reset the knowledge graph"""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def main():
    try:
        from mcp.server.stdio import stdio_server

        logger.info("Starting Cognee MCP server...")

        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options=InitializationOptions(
                    server_name="cognee",
                    server_version="0.1.0",
                    capabilities=mcp.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
                raise_exceptions=True,
            )

    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}", exc_info=True)
        raise


# async def visualize() -> Image:
#     """Visualize the knowledge graph"""
#     try:
#         image_path = await cognee.visualize_graph()

#         img = PILImage.open(image_path)
#         return Image(data=img.tobytes(), format="png")
#     except (FileNotFoundError, IOError, ValueError) as e:
#       raise ValueError(f"Failed to create visualization: {str(e)}")


def node_to_string(node):
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )

    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    edge_strings = []
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship_type = edge["relationship_name"]
        edge_str = f"{node_to_string(node1)} {relationship_type} {node_to_string(node2)}"
        edge_strings.append(edge_str)
    return "\n".join(edge_strings)


def load_class(model_file, model_name):
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, model_name)

    return model_class


# def get_freshest_png(directory: str) -> Image:
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f"Directory {directory} does not exist")

#     # List all files in 'directory' that end with .png
#     files = [f for f in os.listdir(directory) if f.endswith(".png")]
#     if not files:
#         raise FileNotFoundError("No PNG files found in the given directory.")

#     # Sort by integer value of the filename (minus the '.png')
#     # Example filename: 1673185134.png -> integer 1673185134
#     try:
#         files_sorted = sorted(files, key=lambda x: int(x.replace(".png", "")))
#     except ValueError as e:
#         raise ValueError("Invalid PNG filename format. Expected timestamp format.") from e

#     # The "freshest" file has the largest timestamp
#     freshest_filename = files_sorted[-1]
#     freshest_path = os.path.join(directory, freshest_filename)

#     # Open the image with PIL and return the PIL Image object
#     try:
#         return PILImage.open(freshest_path)
#     except (IOError, OSError) as e:
#         raise IOError(f"Failed to open PNG file {freshest_path}") from e

if __name__ == "__main__":
    # Initialize and run the server
    asyncio.run(main())
