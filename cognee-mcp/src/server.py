import json
import os
import sys
import argparse
import cognee
import asyncio
from cognee.shared.logging_utils import get_logger, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.methods import get_default_user
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.storage.utils import JSONEncoder

mcp = FastMCP("Cognee")

logger = get_logger()
log_file = get_log_file_location()


@mcp.tool()
async def cognify(text: str, graph_model_file: str = None, graph_model_name: str = None) -> list:
    async def cognify_task(
        text: str, graph_model_file: str = None, graph_model_name: str = None
    ) -> str:
        """Build knowledge graph from the input text"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Cognify process starting.")
            if graph_model_file and graph_model_name:
                graph_model = load_class(graph_model_file, graph_model_name)
            else:
                graph_model = KnowledgeGraph

            await cognee.add(text)

            try:
                await cognee.cognify(graph_model=graph_model)
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    asyncio.create_task(
        cognify_task(
            text=text,
            graph_model_file=graph_model_file,
            graph_model_name=graph_model_name,
        )
    )

    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current cognify status use the cognify_status tool\n"
        f"or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def codify(repo_path: str) -> list:
    async def codify_task(repo_path: str):
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Codify process starting.")
            results = []
            async for result in run_code_graph_pipeline(repo_path, False):
                results.append(result)
                logger.info(result)
            if all(results):
                logger.info("Codify process finished succesfully.")
            else:
                logger.info("Codify process failed.")

    asyncio.create_task(codify_task(repo_path))

    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current codify status use the codify_status tool\n"
        f"or you can check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def search(search_query: str, search_type: str) -> list:
    async def search_task(search_query: str, search_type: str) -> str:
        """Search the knowledge graph"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            search_results = await cognee.search(
                query_type=SearchType[search_type.upper()], query_text=search_query
            )

            if search_type.upper() == "CODE":
                return json.dumps(search_results, cls=JSONEncoder)
            elif (
                search_type.upper() == "GRAPH_COMPLETION" or search_type.upper() == "RAG_COMPLETION"
            ):
                return search_results[0]
            elif search_type.upper() == "CHUNKS":
                return str(search_results)
            elif search_type.upper() == "INSIGHTS":
                results = retrieved_edges_to_string(search_results)
                return results
            else:
                return str(search_results)

    search_results = await search_task(search_query, search_type)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
async def prune():
    """Reset the knowledge graph"""
    with redirect_stdout(sys.stderr):
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        return [types.TextContent(type="text", text="Pruned")]


@mcp.tool()
async def cognify_status():
    """Get status of cognify pipeline"""
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status([await get_unique_dataset_id("main_dataset", user)])
        return [types.TextContent(type="text", text=str(status))]


@mcp.tool()
async def codify_status():
    """Get status of codify pipeline"""
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status([await get_unique_dataset_id("codebase", user)])
        return [types.TextContent(type="text", text=str(status))]


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


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(
            f"Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}"
        )
        await mcp.run_sse_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
