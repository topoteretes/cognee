import importlib.util
import os
import asyncio
from contextlib import redirect_stderr, redirect_stdout

import cognee
import mcp.server.stdio
import mcp.types as types
from cognee.api.v1.search import SearchType
from cognee.shared.data_models import KnowledgeGraph
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from PIL import Image
from PIL import Image as PILImage

server = Server("cognee-mcp")


def node_to_string(node):
    # keys_to_keep = ["chunk_index", "topological_rank", "cut_type", "id", "text"]
    # keyset = set(keys_to_keep) & node.keys()
    # return "Node(" + " ".join([key + ": " + str(node[key]) + "," for key in keyset]) + ")"
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


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="cognify",
            description="Build knowledge graph from the input text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "graph_model_file": {"type": "string"},
                    "graph_model_name": {"type": "string"},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="search",
            description="Search the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="prune",
            description="Reset the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="visualize",
            description="Visualize the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        ),
    ]


def get_freshest_png(directory: str) -> Image.Image:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # List all files in 'directory' that end with .png
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    if not files:
        raise FileNotFoundError("No PNG files found in the given directory.")

    # Sort by integer value of the filename (minus the '.png')
    # Example filename: 1673185134.png -> integer 1673185134
    try:
        files_sorted = sorted(files, key=lambda x: int(x.replace(".png", "")))
    except ValueError as e:
        raise ValueError("Invalid PNG filename format. Expected timestamp format.") from e

    # The "freshest" file has the largest timestamp
    freshest_filename = files_sorted[-1]
    freshest_path = os.path.join(directory, freshest_filename)

    # Open the image with PIL and return the PIL Image object
    try:
        return Image.open(freshest_path)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to open PNG file {freshest_path}") from e

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "cognify":
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                if not arguments:
                    raise ValueError("Missing arguments")

                text = arguments.get("text")

                if ("graph_model_file" in arguments) and ("graph_model_name" in arguments):
                    model_file = arguments.get("graph_model_file")
                    model_name = arguments.get("graph_model_name")

                    graph_model = load_class(model_file, model_name)
                else:
                    graph_model = KnowledgeGraph

                await cognee.add(text)

                await cognee.cognify(graph_model=graph_model)

                return [
                    types.TextContent(
                        type="text",
                        text="Ingested",
                    )
                ]
    elif name == "search":
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                if not arguments:
                    raise ValueError("Missing arguments")

                search_query = arguments.get("query")

                search_results = await cognee.search(SearchType.INSIGHTS, query_text=search_query)

                results = retrieved_edges_to_string(search_results)

                return [
                    types.TextContent(
                        type="text",
                        text=results,
                    )
                ]
    elif name == "prune":
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                await cognee.prune.prune_data()
                await cognee.prune.prune_system(metadata=True)

                return [
                    types.TextContent(
                        type="text",
                        text="Pruned",
                    )
                ]

    elif name == "visualize":
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                try:
                    await cognee.visualize()
                    img = get_freshest_png(".")
                    return types.Image(data=img.tobytes(), format="png")
                except (FileNotFoundError, IOError, ValueError) as e:
                    raise ValueError(f"Failed to create visualization: {str(e)}")
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cognee-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# This is needed if you'd like to connect to a custom client
if __name__ == "__main__":
    asyncio.run(main())
