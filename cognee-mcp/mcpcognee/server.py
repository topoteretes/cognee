import importlib.util
import os
from contextlib import redirect_stderr, redirect_stdout

import cognee
import mcp.server.stdio
import mcp.types as types
from cognee.api.v1.search import SearchType
from cognee.shared.data_models import KnowledgeGraph
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl, BaseModel

server = Server("mcpcognee")


def node_to_string(node):
    keys_to_keep = ["chunk_index", "topological_rank", "cut_type", "id", "text"]
    keyset = set(keys_to_keep) & node.keys()
    return "Node(" + " ".join([key + ": " + str(node[key]) + "," for key in keyset]) + ")"


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
            name="Cognify_and_search",
            description="Build knowledge graph from the input text and search in it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "search_query": {"type": "string"},
                    "graph_model_file": {"type": "string"},
                    "graph_model_name": {"type": "string"},
                },
                "required": ["text", "search_query"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "Cognify_and_search":
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                await cognee.prune.prune_data()
                await cognee.prune.prune_system(metadata=True)

                if not arguments:
                    raise ValueError("Missing arguments")

                text = arguments.get("text")
                search_query = arguments.get("search_query")
                if ("graph_model_file" in arguments) and ("graph_model_name" in arguments):
                    model_file = arguments.get("graph_model_file")
                    model_name = arguments.get("graph_model_name")
                    graph_model = load_class(model_file, model_name)
                else:
                    graph_model = KnowledgeGraph

                await cognee.add(text)
                await cognee.cognify(graph_model=graph_model)
                search_results = await cognee.search(SearchType.INSIGHTS, query_text=search_query)

                results = retrieved_edges_to_string(search_results)

                return [
                    types.TextContent(
                        type="text",
                        text=results,
                    )
                ]
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcpcognee",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
